"""Run the full batch pipeline + evaluation on the synthetic fixture with NO API keys.

Deepgram and pyannote are replaced by their cached outputs (tests/fixtures), the LLM by the
rule-based HeuristicLLM. Everything else — alignment, overlap flags, utterances, map-reduce,
grounding, WER/cpWER/DER, report — is the real code.

  python scripts/demo_offline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convintel.data.ami import load_reference  # noqa: E402
from convintel.diarization.pyannote_diar import DiarizationResult, make_exclusive, read_rttm  # noqa: E402
from convintel.evals.runner import aggregate, evaluate_meeting, write_report  # noqa: E402
from convintel.pipeline import BatchPipeline, PipelineOptions  # noqa: E402
from convintel.summarize import MeetingExtractor  # noqa: E402
from convintel.summarize.heuristic import HeuristicLLM  # noqa: E402

FX = ROOT / "tests" / "fixtures"


class CachedASR:
    def transcribe_file(self, path, **_):
        return json.loads((FX / "deepgram_FX001a.json").read_text())


class CachedDiarizer:
    def __call__(self, path, **_):
        turns = read_rttm(FX / "diarization_FX001a.rttm")
        return DiarizationResult(turns, make_exclusive(turns), model="fixture-rttm")


def main(out_dir: Path = ROOT / "outputs" / "demo"):
    pipe = BatchPipeline(CachedASR(), CachedDiarizer(), MeetingExtractor(HeuristicLLM()), output_dir=out_dir)
    res = pipe.run(FX / "FX001a.wav", meeting_id="FX001a", opts=PipelineOptions(use_cache=False))
    print("=== transcript ===")
    print(res.transcript.as_prompt_text())
    print("\n=== record ===")
    print((res.out_dir / "record.md").read_text())

    ref = load_reference("FX001a", FX / "ami")
    row = evaluate_meeting(ref, res.transcript, res.record, res.diarization.turns)
    path = write_report([row], aggregate([row]), out_dir / "eval")
    print("=== metrics ===")
    print(json.dumps({"wer": row["asr"]["wer"], "cpwer": row["asr"]["cpwer"],
                      "der_strict": row["der"]["strict"]["der"],
                      "der_breakdown": {k: row["der"]["strict"][k] for k in ("miss", "false_alarm", "confusion")}},
                     indent=1))
    print(f"report: {path}")
    return res, row


if __name__ == "__main__":
    main()
