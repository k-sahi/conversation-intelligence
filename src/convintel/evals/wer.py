"""Word error rate (WER) and concatenated minimum-permutation WER (cpWER).

WER   = (S + D + I) / N_ref over the whole meeting in time order. Measures ASR only.
cpWER = concatenate each speaker's words into one stream (reference and hypothesis),
        find the speaker mapping that minimises total errors, report WER under it.
        Penalises both recognition errors AND words given to the wrong speaker,
        so it is the single number for "who said what" quality (CHiME-6 / NOTSOFAR metric).

Normalisation is applied symmetrically to reference and hypothesis. Without it,
"OK" vs "okay", "5" vs "five", and fillers dominate the error count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FILLERS = {"um", "uh", "hmm", "mm", "mhm", "mm-hmm", "uh-huh", "ah", "eh", "er", "erm", "hm", "uhm", "huh"}
EQUIV = {"ok": "okay", "'kay": "okay", "kay": "okay", "alright": "all right", "yep": "yeah", "yup": "yeah",
         "gonna": "going to", "wanna": "want to", "gotta": "got to", "cause": "because", "'cause": "because"}
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _num_to_words(rest))
    if n < 10000:
        rest = n % 1000
        return _num_to_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + _num_to_words(rest))
    return " ".join(_ONES[int(d)] for d in str(n))


def normalize(text: str, drop_fillers: bool = True) -> list[str]:
    t = text.lower()
    t = t.replace("%", " percent").replace("&", " and ").replace("€", " euro ").replace("$", " dollar ")
    t = re.sub(r"(\d+)\.(\d+)", lambda m: f"{m.group(1)} point {' '.join(m.group(2))}", t)
    t = re.sub(r"\d+", lambda m: f" {_num_to_words(int(m.group()))} ", t)
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"[^a-z' ]+", " ", t)
    out: list[str] = []
    for tok in t.split():
        tok = tok.strip("'")
        if not tok:
            continue
        tok = EQUIV.get(tok, tok)
        for piece in tok.split():
            if drop_fillers and piece in FILLERS:
                continue
            out.append(piece)
    return out


@dataclass
class EditCounts:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def wer(self) -> float:
        return self.errors / self.ref_len if self.ref_len else float(self.ins > 0)

    def __add__(self, o: "EditCounts") -> "EditCounts":
        return EditCounts(self.sub + o.sub, self.dele + o.dele, self.ins + o.ins, self.ref_len + o.ref_len)

    def as_dict(self) -> dict:
        return {"wer": round(self.wer, 4), "sub": self.sub, "del": self.dele, "ins": self.ins, "ref_words": self.ref_len}


def edit_counts(ref: list[str], hyp: list[str]) -> EditCounts:
    """Levenshtein alignment with S/D/I breakdown. Uses rapidfuzz if installed (C speed),
    otherwise an O(len(hyp)) memory pure-Python DP that carries counts per cell."""
    try:
        from rapidfuzz.distance import Levenshtein

        vocab: dict[str, int] = {}
        r = [vocab.setdefault(w, len(vocab)) for w in ref]
        h = [vocab.setdefault(w, len(vocab)) for w in hyp]
        c = EditCounts(ref_len=len(ref))
        for op in Levenshtein.editops(r, h):
            if op.tag == "replace":
                c.sub += 1
            elif op.tag == "delete":
                c.dele += 1
            elif op.tag == "insert":
                c.ins += 1
        return c
    except ImportError:
        pass

    m = len(hyp)
    # each cell: (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)] + [None] * m  # type: ignore[list-item]
        rw = ref[i - 1]
        for j in range(1, m + 1):
            d = prev[j - 1]
            if rw == hyp[j - 1]:
                best = d
            else:
                best = (d[0] + 1, d[1] + 1, d[2], d[3])
            u = prev[j]
            if u[0] + 1 < best[0]:
                best = (u[0] + 1, u[1], u[2] + 1, u[3])
            left = cur[j - 1]
            if left[0] + 1 < best[0]:
                best = (left[0] + 1, left[1], left[2], left[3] + 1)
            cur[j] = best
        prev = cur
    _, s, dl, ins = prev[m]
    return EditCounts(s, dl, ins, len(ref))


def wer(ref_text: str, hyp_text: str) -> EditCounts:
    return edit_counts(normalize(ref_text), normalize(hyp_text))


def cpwer(ref_by_spk: dict[str, str], hyp_by_spk: dict[str, str]) -> tuple[EditCounts, dict[str, str]]:
    """Returns (counts, mapping ref_speaker -> hyp_speaker). Unmatched streams count as
    all-deletions (missing speaker) or all-insertions (extra speaker)."""
    refs = {k: normalize(v) for k, v in ref_by_spk.items()}
    hyps = {k: normalize(v) for k, v in hyp_by_spk.items()}
    rk, hk = list(refs), list(hyps)
    n = max(len(rk), len(hk))
    rk += [f"__pad_r{i}" for i in range(n - len(rk))]
    hk += [f"__pad_h{i}" for i in range(n - len(hk))]
    cost = [[edit_counts(refs.get(r, []), hyps.get(h, [])) for h in hk] for r in rk]
    rows, cols = _assign([[c.errors for c in row] for row in cost])
    total = EditCounts()
    mapping = {}
    for r, c in zip(rows, cols):
        total = total + cost[r][c]
        if not rk[r].startswith("__pad") and not hk[c].startswith("__pad"):
            mapping[rk[r]] = hk[c]
    return total, mapping


def _assign(cost: list[list[float]]) -> tuple[list[int], list[int]]:
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        return list(map(int, r)), list(map(int, c))
    except ImportError:  # brute force for small n
        import itertools

        n = len(cost)
        best = min(itertools.permutations(range(n)), key=lambda p: sum(cost[i][p[i]] for i in range(n)))
        return list(range(n)), list(best)
