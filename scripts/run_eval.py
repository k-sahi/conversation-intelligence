"""Evaluate the batch system on AMI.

  python scripts/run_eval.py --split dev-mini                    # full: ASR + diarization + summary + judge
  python scripts/run_eval.py --split test --no-judge             # WER/cpWER/DER only (cheap)
  python scripts/run_eval.py --split test --oracle-speakers      # report separately: uses true speaker count

Results -> outputs/eval/<run-name>/{per_meeting.jsonl, aggregate.json, report.md}
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convintel.asr.deepgram_batch import AMI_KEYTERMS  # noqa: E402
from convintel.config import settings  # noqa: E402
from convintel.data.ami import download_annotations, download_audio, load_reference, meetings_in  # noqa: E402
from convintel.evals.judge import ItemMatcher, RubricJudge  # noqa: E402
from convintel.evals.runner import aggregate, evaluate_meeting, write_report  # noqa: E402
from convintel.pipeline import BatchPipeline, PipelineOptions  # noqa: E402
from convintel.summarize.llm import AnthropicLLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev-mini")
    ap.add_argument("--stream", default="Mix-Headset")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--oracle-speakers", action="store_true")
    ap.add_argument("--no-keyterms", action="store_true")
    ap.add_argument("--judge-samples", type=int, default=1)
    ap.add_argument("--use-pyannote-metrics", action="store_true")
    a = ap.parse_args()

    run = a.run_name or f"{a.split}-{a.stream}-{time.strftime('%Y%m%d-%H%M')}"
    ann = download_annotations(settings.data_dir)
    pipe = BatchPipeline(output_dir=settings.output_dir / "runs" / a.stream)
    judge = matcher = None
    if not (a.no_judge or a.no_summary):
        jllm = AnthropicLLM(model=settings.judge_model)
        judge, matcher = RubricJudge(jllm, n_samples=a.judge_samples), ItemMatcher(jllm)

    rows = []
    for m in meetings_in(a.split):
        ref = load_reference(m, ann)
        audio = download_audio(m, settings.data_dir, a.stream)
        opts = PipelineOptions(num_speakers=len(ref.speakers) if a.oracle_speakers else None,
                               keyterms=None if a.no_keyterms else AMI_KEYTERMS, skip_summary=a.no_summary)
        t = time.perf_counter()
        res = pipe.run(audio, meeting_id=m, opts=opts)
        hyp_turns = res.diarization.turns if res.diarization else None
        row = evaluate_meeting(ref, res.transcript, res.record, hyp_turns, judge, matcher, a.use_pyannote_metrics)
        row["timings"] = res.timings
        row["wall_s"] = round(time.perf_counter() - t, 1)
        rows.append(row)
        print(json.dumps({"meeting": m, "wer": row["asr"]["wer"]["wer"], "cpwer": row["asr"]["cpwer"]["wer"],
                          "der": row.get("der", {}).get("strict", {}).get("der")}))

    agg = aggregate(rows)
    agg["config"] = {"split": a.split, "stream": a.stream, "oracle_speakers": a.oracle_speakers,
                     "asr": settings.deepgram_model, "diarization": settings.pyannote_pipeline,
                     "extraction_llm": settings.extraction_model, "judge_llm": settings.judge_model}
    path = write_report(rows, agg, settings.output_dir / "eval" / run)
    print(f"\nreport: {path}\n{json.dumps(agg, indent=1)}")


if __name__ == "__main__":
    main()
