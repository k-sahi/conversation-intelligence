"""Calibrating the LLM judge against human ratings.

A judge is only useful if its scores track human judgement. Protocol (docs/evaluation.md):
  1. Two annotators score ~30 meeting records on the rubric -> human_labels.jsonl
  2. Measure inter-annotator agreement = the ceiling the judge can hope to reach.
  3. Run the judge on the same records -> judge_labels.jsonl
  4. Per dimension compute quadratic-weighted kappa (QWK), Spearman rho, exact and
     within-one agreement, and mean bias (judge - human).
  5. Iterate on rubric wording / anchors / few-shot examples on a DEV half, report on the
     held-out half. Gate: QWK >= 0.6 and |bias| <= 0.5 on every dimension.

QWK is the headline: it is chance-corrected and penalises a 1-vs-5 disagreement much more
than 4-vs-5, which matches how ordinal rubric scores should be compared.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def quadratic_weighted_kappa(a: list[int], b: list[int], lo: int = 1, hi: int = 5) -> float:
    assert len(a) == len(b) and a, "need paired, non-empty ratings"
    k = hi - lo + 1
    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[int(round(x)) - lo][int(round(y)) - lo] += 1
    n = len(a)
    ha = [sum(O[i]) for i in range(k)]
    hb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * O[i][j]
            den += w * ha[i] * hb[j] / n
    return 1.0 - num / den if den else 1.0


def _ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 0.0


def agreement_report(human: list[float], judge: list[float]) -> dict:
    n = len(human)
    return {
        "n": n,
        "qwk": round(quadratic_weighted_kappa([round(h) for h in human], [round(j) for j in judge]), 3),
        "spearman": round(spearman(human, judge), 3),
        "exact": round(sum(round(h) == round(j) for h, j in zip(human, judge)) / n, 3),
        "within_1": round(sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / n, 3),
        "bias": round(sum(j - h for h, j in zip(human, judge)) / n, 3),
    }


def load_labels(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def calibrate(human_rows: list[dict], judge_rows: list[dict], qwk_gate: float = 0.6, bias_gate: float = 0.5) -> dict:
    """Rows: {"item_id", "dimension", "score", "annotator"(human only)}.
    Human score per (item, dim) = mean over annotators."""
    hum: dict[tuple, list[float]] = defaultdict(list)
    per_annot: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in human_rows:
        key = (r["item_id"], r["dimension"])
        hum[key].append(float(r["score"]))
        per_annot[key][r.get("annotator", "h")] = float(r["score"])
    jud = {(r["item_id"], r["dimension"]): float(r["score"]) for r in judge_rows}

    report: dict = {"dimensions": {}, "passed": True}
    for dim in sorted({k[1] for k in hum}):
        keys = [k for k in hum if k[1] == dim and k in jud]
        if len(keys) < 2:
            continue
        h = [sum(hum[k]) / len(hum[k]) for k in keys]
        j = [jud[k] for k in keys]
        rep = agreement_report(h, j)
        # inter-annotator ceiling, if two annotators labelled the same items
        pairs = [tuple(per_annot[k].values())[:2] for k in keys if len(per_annot[k]) >= 2]
        if len(pairs) >= 2:
            rep["inter_annotator_qwk"] = round(
                quadratic_weighted_kappa([int(p[0]) for p in pairs], [int(p[1]) for p in pairs]), 3)
        rep["passed"] = rep["qwk"] >= qwk_gate and abs(rep["bias"]) <= bias_gate
        report["passed"] &= rep["passed"]
        report["dimensions"][dim] = rep
    return report
