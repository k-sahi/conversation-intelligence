"""End-to-end evaluation over AMI meetings. Produces one JSON row per meeting and an
aggregate report. ASR/diarization/summary stages are cached by the pipeline, so
re-running only the judge (e.g. after changing the rubric) costs only judge calls."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from ..config import settings
from ..data.ami import AMIReference
from ..observability import span
from ..schemas import MeetingRecord, Transcript
from .der import diarization_error_rate
from .judge import ItemMatcher, RubricJudge
from .text_metrics import compression_ratio, rouge_l
from .wer import cpwer, wer


def asr_metrics(ref: AMIReference, transcript: Transcript) -> dict:
    hyp_text = " ".join(u.text for u in sorted(transcript.utterances, key=lambda u: u.start))
    hyp_by_spk: dict[str, list[str]] = {}
    for u in transcript.utterances:
        hyp_by_spk.setdefault(u.speaker, []).append(u.text)
    w = wer(ref.text(), hyp_text)
    cp, mapping = cpwer(ref.text_by_speaker(), {k: " ".join(v) for k, v in hyp_by_spk.items()})
    return {"wer": w.as_dict(), "cpwer": cp.as_dict(), "cpwer_mapping": mapping}


def der_metrics(ref: AMIReference, hyp_turns, use_pyannote_metrics: bool = False) -> dict:
    ref_turns = ref.turns()
    out = {}
    for name, collar, skip in [("strict", 0.0, False), ("collar_0.25", 0.25, False), ("no_overlap", 0.0, True)]:
        if use_pyannote_metrics:
            out[name] = _pyannote_der(ref_turns, hyp_turns, collar, skip)
        else:
            out[name] = diarization_error_rate(ref_turns, hyp_turns, collar=collar, skip_overlap=skip).as_dict()
    return out


def _pyannote_der(ref_turns, hyp_turns, collar, skip):  # pragma: no cover - optional dep
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def ann(turns):
        a = Annotation()
        for t in turns:
            a[Segment(t.start, t.end)] = t.speaker
        return a

    metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip)  # pyannote collar = total width
    d = metric(ann(ref_turns), ann(hyp_turns), detailed=True)
    total = d["total"] or 1.0
    return {"der": round(d["diarization error rate"], 4), "miss": round(d["missed detection"] / total, 4),
            "false_alarm": round(d["false alarm"] / total, 4), "confusion": round(d["confusion"] / total, 4)}


def summary_metrics(ref: AMIReference, record: MeetingRecord, transcript: Transcript,
                    judge: RubricJudge | None, matcher: ItemMatcher | None) -> dict:
    out: dict = {
        "rouge_l_vs_abstract": rouge_l(" ".join(ref.abstract), record.summary),
        "compression_ratio": compression_ratio(sum(len(u.text.split()) for u in transcript.utterances), record.summary),
        "n_decisions": len(record.decisions), "n_actions": len(record.action_items),
        "n_dropped_ungrounded": len(record.warnings),
    }
    if not ref.has_summary():
        out["note"] = "no AMI abstractive summary for this meeting; judge skipped"
        return out
    if matcher:
        out["decision_match"] = matcher.match(ref.decisions, [d.statement for d in record.decisions])
        out["action_match"] = matcher.match(ref.actions, [a.task + (f" ({a.owner})" if a.owner else "")
                                                           for a in record.action_items])
    if judge:
        out["judge"] = judge.score(record, transcript, ref.abstract, ref.decisions, ref.actions)
    return out


def aggregate(rows: list[dict]) -> dict:
    def collect(path: list[str]) -> list[float]:
        vals = []
        for r in rows:
            v = r
            for p in path:
                v = v.get(p) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def stat(path):
        v = collect(path)
        return {"mean": round(statistics.mean(v), 4), "n": len(v)} if v else None

    # Corpus-level WER/DER should be micro-averaged (sum errors / sum ref), not mean of per-meeting rates.
    def micro(kind):
        errs = sum(r["asr"][kind]["sub"] + r["asr"][kind]["del"] + r["asr"][kind]["ins"] for r in rows if "asr" in r)
        n = sum(r["asr"][kind]["ref_words"] for r in rows if "asr" in r)
        return round(errs / n, 4) if n else None

    def micro_der(cond):
        num = den = 0.0
        for r in rows:
            d = r.get("der", {}).get(cond)
            if d and "ref_speech_s" in d:
                num += d["der"] * d["ref_speech_s"]
                den += d["ref_speech_s"]
        return round(num / den, 4) if den else stat(["der", cond, "der"])

    agg = {
        "meetings": len(rows),
        "wer_micro": micro("wer"), "cpwer_micro": micro("cpwer"),
        "der_strict": micro_der("strict"), "der_collar_0.25": micro_der("collar_0.25"),
        "der_no_overlap": micro_der("no_overlap"),
        "decision_recall": stat(["summary", "decision_match", "recall"]),
        "decision_precision": stat(["summary", "decision_match", "precision"]),
        "action_recall": stat(["summary", "action_match", "recall"]),
        "rouge_l_f": stat(["summary", "rouge_l_vs_abstract", "f"]),
    }
    for dim in ["faithfulness", "decision_coverage", "action_item_quality", "summary_usefulness"]:
        agg[f"judge_{dim}"] = stat(["summary", "judge", "scores", dim])
    return agg


def evaluate_meeting(ref: AMIReference, transcript: Transcript, record: MeetingRecord | None,
                     hyp_turns, judge=None, matcher=None, use_pyannote_metrics=False) -> dict:
    with span("eval", meeting_id=ref.meeting) as s:
        row: dict = {"meeting": ref.meeting, "asr": asr_metrics(ref, transcript)}
        if hyp_turns:
            row["der"] = der_metrics(ref, hyp_turns, use_pyannote_metrics)
        if record is not None:
            row["summary"] = summary_metrics(ref, record, transcript, judge, matcher)
        s.set_attribute("eval.wer", row["asr"]["wer"]["wer"])
        s.set_attribute("eval.cpwer", row["asr"]["cpwer"]["wer"])
        if "der" in row:
            s.set_attribute("eval.der", row["der"]["strict"]["der"])
        for dim, v in row.get("summary", {}).get("judge", {}).get("scores", {}).items():
            s.set_attribute(f"eval.judge.{dim}", v)
    return row


def write_report(rows: list[dict], agg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_meeting.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=1))
    lines = ["# Evaluation report", "", "| Metric | Value |", "|---|---|"]
    for k, v in agg.items():
        lines.append(f"| {k} | {v['mean'] if isinstance(v, dict) else v} |")
    lines += ["", "## Per meeting", "", "| Meeting | WER | cpWER | DER (strict) | Dec. recall | Faithfulness |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        s = r.get("summary", {})
        lines.append("| {m} | {w} | {cp} | {d} | {dr} | {f} |".format(
            m=r["meeting"], w=r["asr"]["wer"]["wer"], cp=r["asr"]["cpwer"]["wer"],
            d=r.get("der", {}).get("strict", {}).get("der", "—"),
            dr=s.get("decision_match", {}).get("recall", "—"),
            f=s.get("judge", {}).get("scores", {}).get("faithfulness", "—")))
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


__all__ = ["aggregate", "evaluate_meeting", "write_report", "settings"]
