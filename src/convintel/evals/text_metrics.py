"""Cheap lexical metrics, reported only as sanity checks.

ROUGE-L against the single AMI abstract is known to correlate weakly with human judgement
for abstractive meeting summaries (many valid summaries share few n-grams with one
reference). We log it so a sudden collapse is visible, never to rank systems.
"""
from __future__ import annotations

from .wer import normalize


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, candidate: str) -> dict:
    r, c = normalize(reference), normalize(candidate)
    if not r or not c:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    l = _lcs(r, c)
    p, rec = l / len(c), l / len(r)
    f = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {"p": round(p, 4), "r": round(rec, 4), "f": round(f, 4)}


def compression_ratio(transcript_words: int, summary: str) -> float:
    n = len(summary.split())
    return round(transcript_words / n, 1) if n else float("inf")
