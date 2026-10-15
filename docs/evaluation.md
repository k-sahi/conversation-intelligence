# Evaluation

## Metrics at a glance

| Metric | Measures | Formula | Code |
|---|---|---|---|
| **WER** | ASR accuracy | (S + D + I) / N_ref, whole meeting in time order, after normalisation | `evals/wer.py` |
| **cpWER** | ASR + speaker attribution | WER on per-speaker word streams under the error-minimising speaker mapping (Hungarian) | `evals/wer.py` |
| **DER** | diarization | (miss + false alarm + confusion) / reference speech time; Hungarian ref→hyp mapping | `evals/der.py` |
| Decision / action **P/R/F1** | content coverage | LLM matcher vs AMI `abssumm` decisions / actions | `evals/judge.py` `ItemMatcher` |
| **Rubric judge** (1–5 ×4) | semantic quality | calibrated LLM judge, anchored rubric | `evals/judge.py` `RubricJudge` |
| Pairwise win rate | A vs B | both orders, inconsistent → tie | `PairwiseJudge` |
| **TTFO / TTFW / WPS / TTFA** | real-time latency | see `docs/realtime.md` | `evals/latency.py` |
| ROUGE-L, compression | sanity only | LCS F-measure vs AMI abstract | `evals/text_metrics.py` |

### Normalisation (applies to both reference and hypothesis)
lower-case; digits → words (`25` → `twenty five`); `%`, `&` spelled out; hyphens split; punctuation removed;
fillers removed (`um, uh, mm-hmm, …`); equivalence map (`ok→okay`, `gonna→going to`, …). Without this,
formatting differences between Nova-3 smart-format output and AMI's orthography dominate the error count.

### Aggregation
Corpus WER/cpWER/DER are **micro-averaged** (total errors / total reference), not the mean of per-meeting
rates, so long meetings weigh proportionally. Judge scores are means over meetings with a reference summary.

### DER conditions
* `strict` — no collar, overlap scored (comparable to the pyannote model card's AMI numbers)
* `collar_0.25` — 0.25 s forgiveness around reference boundaries (NIST-style)
* `no_overlap` — overlapped reference frames excluded; `strict − no_overlap` ≈ cost of overlap

Our DER implementation is frame-based (10 ms) and dependency-free for tests;
`--use-pyannote-metrics` scores with `pyannote.metrics` for reported numbers.

## Protocol

1. **Data**: standard AMI full-corpus-ASR partition. Iterate on `dev-mini` (3 meetings) and `dev`; report on
   `test` once. Summary metrics only on meetings with an abstractive summary (`test-scenario`).
2. **Conditions**: Mix-Headset (close-talk mix) and Array1-01 (single distant mic). Blind speaker count by
   default; oracle-count run reported separately.
3. **Caching**: ASR and diarization are cached per meeting; prompt or rubric changes only re-run LLM stages.
4. **Run**: `python scripts/run_eval.py --split test-scenario --stream Mix-Headset`.
5. **Outputs**: `outputs/eval/<run>/per_meeting.jsonl`, `aggregate.json`, `report.md`; spans + `eval.*`
   attributes in Arize.

## Judge calibration

```
python scripts/calibrate_judge.py sheet  --records outputs/runs/Mix-Headset --out calibration/sheet.csv
#   two annotators fill the sheet using docs/rubric.md  -> calibration/human_labels.jsonl
python scripts/calibrate_judge.py judge  --records outputs/runs/Mix-Headset --out calibration/judge_labels.jsonl
python scripts/calibrate_judge.py report --human calibration/human_labels.jsonl --judge calibration/judge_labels.jsonl
```

Report per dimension: **quadratic-weighted kappa** (headline: chance-corrected, ordinal-aware),
Spearman ρ, exact and within-1 agreement, mean bias (judge − human), and inter-annotator QWK (the ceiling).

**Gate**: QWK ≥ 0.6 and |bias| ≤ 0.5 on every dimension on the held-out half. If a dimension fails:
1. read the disagreements — usually an ambiguous anchor;
2. rewrite that anchor, add one worked example from the dev half;
3. re-run on dev; touch the held-out half only once per rubric version.

Additional checks: judge score spread across `n_samples=3` (instability), pairwise order consistency
(position bias), and the heuristic-baseline control (must score clearly lower).

## Recording results

Paste `aggregate.json` values into the README results table. Always state: split, audio stream,
blind vs oracle speakers, model versions (stored in `aggregate.json["config"]`), and the date.
