"""Diarization error rate (DER), frame-based, overlap-aware.

DER = (missed speech + false alarm + speaker confusion) / total reference speech time

Per 10 ms frame with N_ref reference speakers and N_hyp hypothesis speakers active:
  miss      = max(0, N_ref - N_hyp)
  false_al. = max(0, N_hyp - N_ref)
  confusion = min(N_ref, N_hyp) - N_correct
where N_correct counts reference speakers whose mapped hypothesis speaker is also active.
The ref->hyp speaker mapping is the one that maximises total co-occurrence (Hungarian).

Overlapped frames count once per reference speaker, so a system that outputs only one
speaker during overlap is charged a miss — the reason exclusive diarization alone is
not enough for scoring and why we keep the overlapping view.

Options mirror common evaluation setups:
  collar        – seconds ignored around each reference boundary (0.0 = strict, 0.25 = NIST)
  skip_overlap  – ignore frames where the reference has 2+ speakers
This is a dependency-free reference implementation used in unit tests and quick
checks. For reported numbers, `scripts/run_eval.py --use-pyannote-metrics` scores with
pyannote.metrics, the tool used by the published AMI baselines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import SpeakerTurn


@dataclass
class DERResult:
    der: float
    miss: float
    false_alarm: float
    confusion: float
    total: float
    mapping: dict[str, str]
    overlap_share: float           # fraction of reference speech that is overlapped

    def as_dict(self) -> dict:
        t = self.total or 1.0
        return {
            "der": round(self.der, 4),
            "miss": round(self.miss / t, 4),
            "false_alarm": round(self.false_alarm / t, 4),
            "confusion": round(self.confusion / t, 4),
            "ref_speech_s": round(self.total, 2),
            "overlap_share": round(self.overlap_share, 4),
            "mapping": self.mapping,
        }


def _matrix(turns: list[SpeakerTurn], labels: list[str], n: int, step: float) -> np.ndarray:
    m = np.zeros((len(labels), n), dtype=bool)
    idx = {s: i for i, s in enumerate(labels)}
    for t in turns:
        a, b = int(round(t.start / step)), int(round(t.end / step))
        m[idx[t.speaker], max(0, a):min(n, b)] = True
    return m


def diarization_error_rate(
    reference: list[SpeakerTurn],
    hypothesis: list[SpeakerTurn],
    collar: float = 0.0,
    skip_overlap: bool = False,
    step: float = 0.01,
    uem: tuple[float, float] | None = None,
) -> DERResult:
    end = max([t.end for t in reference + hypothesis] + [0.0])
    n = int(np.ceil(end / step)) + 1
    rl = sorted({t.speaker for t in reference})
    hl = sorted({t.speaker for t in hypothesis})
    R = _matrix(reference, rl, n, step)
    H = _matrix(hypothesis, hl, n, step) if hl else np.zeros((0, n), dtype=bool)

    scored = np.ones(n, dtype=bool)
    if uem:
        scored[: int(uem[0] / step)] = False
        scored[int(uem[1] / step):] = False
    if collar > 0:
        c = int(round(collar / step))
        for t in reference:
            for b in (t.start, t.end):
                k = int(round(b / step))
                scored[max(0, k - c): k + c] = False
    n_ref = R.sum(0)
    if skip_overlap:
        scored &= n_ref <= 1

    R, H = R[:, scored], H[:, scored]
    n_ref, n_hyp = R.sum(0), H.sum(0)

    # optimal mapping on co-occurrence
    mapping: dict[str, str] = {}
    n_correct = np.zeros(R.shape[1], dtype=int)
    if rl and hl:
        co = R.astype(np.int64) @ H.T.astype(np.int64)
        try:
            from scipy.optimize import linear_sum_assignment

            ri, hi = linear_sum_assignment(-co)
        except ImportError:  # greedy fallback
            ri, hi, used = [], [], set()
            for r in np.argsort(-co.max(1)):
                h = next((h for h in np.argsort(-co[r]) if h not in used), None)
                if h is not None:
                    ri.append(r); hi.append(h); used.add(h)
        for r, h in zip(ri, hi):
            if co[r, h] > 0:
                mapping[rl[r]] = hl[h]
                n_correct += R[r] & H[h]

    miss = np.maximum(0, n_ref - n_hyp).sum() * step
    fa = np.maximum(0, n_hyp - n_ref).sum() * step
    conf = (np.minimum(n_ref, n_hyp) - n_correct).sum() * step
    total = n_ref.sum() * step
    ov = (n_ref[n_ref > 1]).sum() * step
    der = (miss + fa + conf) / total if total else 0.0
    return DERResult(float(der), float(miss), float(fa), float(conf), float(total), mapping,
                     float(ov / total) if total else 0.0)
