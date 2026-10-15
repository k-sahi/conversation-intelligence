"""Calibrate the LLM judge against human ratings.

Step 1 – make an annotation sheet from evaluated records:
  python scripts/calibrate_judge.py sheet --records outputs/runs/Mix-Headset --out calibration/sheet.csv
Step 2 – two annotators fill the sheet (score 1-5 per dimension), export to
         calibration/human_labels.jsonl  ({"item_id","dimension","score","annotator"} per line)
Step 3 – score the same items with the judge and compare:
  python scripts/calibrate_judge.py judge --records outputs/runs/Mix-Headset --out calibration/judge_labels.jsonl
  python scripts/calibrate_judge.py report --human calibration/human_labels.jsonl --judge calibration/judge_labels.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convintel.config import settings  # noqa: E402
from convintel.data.ami import download_annotations, load_reference  # noqa: E402
from convintel.evals.calibration import calibrate, load_labels  # noqa: E402
from convintel.evals.judge import RubricJudge  # noqa: E402
from convintel.evals.rubric import DIMENSIONS  # noqa: E402
from convintel.schemas import MeetingRecord, Transcript  # noqa: E402
from convintel.summarize.llm import AnthropicLLM  # noqa: E402


def _items(records_dir: Path):
    for d in sorted(p for p in records_dir.iterdir() if (p / "record.json").exists()):
        yield d.name, MeetingRecord.model_validate_json((d / "record.json").read_text()), \
            Transcript.model_validate_json((d / "transcript.json").read_text())


def cmd_sheet(a):
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "summary", "decisions", "action_items", *DIMENSIONS, "annotator", "notes"])
        for mid, rec, _ in _items(Path(a.records)):
            w.writerow([mid, rec.summary, " | ".join(d.statement for d in rec.decisions),
                        " | ".join(f"{x.task} ({x.owner})" for x in rec.action_items), *[""] * len(DIMENSIONS), "", ""])
    print(f"wrote {a.out}")


def cmd_judge(a):
    ann = download_annotations(settings.data_dir)
    judge = RubricJudge(AnthropicLLM(model=a.model or settings.judge_model), n_samples=a.samples)
    with open(a.out, "w") as fh:
        for mid, rec, tr in _items(Path(a.records)):
            ref = load_reference(mid, ann)
            res = judge.score(rec, tr, ref.abstract, ref.decisions, ref.actions)
            for dim, score in res["scores"].items():
                fh.write(json.dumps({"item_id": mid, "dimension": dim, "score": score}) + "\n")
            print(mid, res["scores"])


def cmd_report(a):
    rep = calibrate(load_labels(Path(a.human)), load_labels(Path(a.judge)), a.qwk_gate, a.bias_gate)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["passed"] else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet"); s.add_argument("--records", required=True); s.add_argument("--out", required=True)
    j = sub.add_parser("judge"); j.add_argument("--records", required=True); j.add_argument("--out", required=True)
    j.add_argument("--model"); j.add_argument("--samples", type=int, default=1)
    r = sub.add_parser("report"); r.add_argument("--human", required=True); r.add_argument("--judge", required=True)
    r.add_argument("--qwk-gate", type=float, default=0.6); r.add_argument("--bias-gate", type=float, default=0.5)
    a = ap.parse_args()
    {"sheet": cmd_sheet, "judge": cmd_judge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
