"""Download AMI annotations and audio for a split.

  python scripts/download_ami.py --split dev-mini --stream Mix-Headset
  python scripts/download_ami.py --split test --stream Array1-01     # far-field SDM
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convintel.config import settings  # noqa: E402
from convintel.data.ami import download_annotations, download_audio, meetings_in  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev-mini", help="dev | test | dev-mini | test-scenario | comma list")
    ap.add_argument("--stream", default="Mix-Headset", help="Mix-Headset | Array1-01 | Headset-0 ...")
    ap.add_argument("--annotations-only", action="store_true")
    a = ap.parse_args()

    ann = download_annotations(settings.data_dir)
    print(f"annotations -> {ann}")
    if a.annotations_only:
        return
    for m in meetings_in(a.split):
        try:
            p = download_audio(m, settings.data_dir, a.stream)
            print(f"{m}: {p} ({p.stat().st_size / 1e6:.0f} MB)")
        except Exception as exc:
            print(f"{m}: FAILED {exc}")


if __name__ == "__main__":
    main()
