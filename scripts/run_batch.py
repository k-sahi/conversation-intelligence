"""Run the batch pipeline on one audio file.

  python scripts/run_batch.py data/audio/ES2011a.Mix-Headset.wav
  python scripts/run_batch.py my_meeting.wav --num-speakers 4 --keyterms "Acme,Q3 roadmap"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convintel.asr.deepgram_batch import AMI_KEYTERMS  # noqa: E402
from convintel.pipeline import BatchPipeline, PipelineOptions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--meeting-id")
    ap.add_argument("--multichannel", action="store_true")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--keyterms", help="comma-separated; 'ami' for the AMI scenario list")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    kt = AMI_KEYTERMS if a.keyterms == "ami" else ([k.strip() for k in a.keyterms.split(",")] if a.keyterms else None)
    res = BatchPipeline().run(a.audio, a.meeting_id, PipelineOptions(
        multichannel=a.multichannel, num_speakers=a.num_speakers, keyterms=kt, use_cache=not a.no_cache))
    print(f"outputs in {res.out_dir}")
    print(f"timings: {res.timings}")
    if res.record:
        print((res.out_dir / "record.md").read_text())


if __name__ == "__main__":
    main()
