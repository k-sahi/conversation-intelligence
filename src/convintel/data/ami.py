"""AMI Meeting Corpus: download, parse NXT annotations, build references.

What we use from AMI (manual annotations v1.6.2, CC BY 4.0):
  words/<M>.<S>.words.xml       word-level times per speaker letter S (A-D) -> WER refs + RTTM refs
  abstractive/<M>.abssumm.xml   human abstract + decisions + actions + problems -> summary refs
Audio:
  <M>.Mix-Headset.wav           IHM mix (all headsets summed)  - "clean-ish" single channel
  <M>.Array1-01.wav             SDM (one far-field array mic)  - hardest, most realistic
  <M>.Headset-<n>.wav           individual headsets           - multichannel / IHM mode

Splits follow the standard "full-corpus-ASR" partition so numbers are comparable
with published results.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..schemas import SpeakerTurn

ANNOT_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{stream}.wav"
NITE = "{http://nite.sourceforge.net/}"


def _four(prefix: str) -> list[str]:
    return [f"{prefix}{c}" for c in "abcd"]


SPLITS: dict[str, list[str]] = {
    "test": _four("ES2004") + _four("IS1009") + _four("TS3003") + _four("TS3007") + _four("EN2002"),
    "dev": _four("ES2011") + _four("IS1008") + _four("TS3004") + _four("TS3006")
           + ["IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"],
}
# A small, cheap subset for iteration: scenario meetings that all have abstractive summaries.
SPLITS["dev-mini"] = ["ES2011a", "IS1008b", "TS3004c"]
SPLITS["test-scenario"] = _four("ES2004") + _four("IS1009") + _four("TS3003") + _four("TS3007")


# ------------------------------------------------------------------ download
def download_annotations(data_dir: Path) -> Path:
    target = data_dir / "ami_annotations"
    if (target / "words").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    r = requests.get(ANNOT_URL, timeout=600)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(target)
    return target


def download_audio(meeting: str, data_dir: Path, stream: str = "Mix-Headset") -> Path:
    out = data_dir / "audio" / f"{meeting}.{stream}.wav"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(AUDIO_URL.format(m=meeting, stream=stream), stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = out.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(out)
    return out


# ------------------------------------------------------------------ parsing
@dataclass
class RefWord:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class AMIReference:
    meeting: str
    words: list[RefWord]
    abstract: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        return sorted({w.speaker for w in self.words})

    def text(self) -> str:
        """Whole-meeting reference in time order (for plain WER)."""
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.start))

    def text_by_speaker(self) -> dict[str, str]:
        """Per-speaker streams (for cpWER)."""
        out: dict[str, list[str]] = {}
        for w in sorted(self.words, key=lambda w: w.start):
            out.setdefault(w.speaker, []).append(w.text)
        return {k: " ".join(v) for k, v in out.items()}

    def turns(self, merge_gap: float = 0.3) -> list[SpeakerTurn]:
        """Reference speaker turns from word timings (overlaps preserved)."""
        turns: list[SpeakerTurn] = []
        for spk in self.speakers:
            cur: SpeakerTurn | None = None
            for w in sorted((w for w in self.words if w.speaker == spk), key=lambda w: w.start):
                if cur and w.start - cur.end <= merge_gap:
                    cur.end = max(cur.end, w.end)
                else:
                    cur = SpeakerTurn(speaker=spk, start=w.start, end=w.end)
                    turns.append(cur)
        return sorted(turns, key=lambda t: t.start)

    def has_summary(self) -> bool:
        return bool(self.abstract)


def parse_words_file(path: Path, speaker: str) -> list[RefWord]:
    words = []
    root = ET.parse(path).getroot()
    for el in root:
        tag = el.tag.split("}")[-1]
        if tag != "w" or el.get("punc") == "true":
            continue
        st, et = el.get("starttime"), el.get("endtime")
        if st is None or et is None or not (el.text or "").strip():
            continue
        words.append(RefWord(speaker=speaker, start=float(st), end=float(et), text=el.text.strip()))
    return words


def parse_abssumm(path: Path) -> dict[str, list[str]]:
    root = ET.parse(path).getroot()
    out: dict[str, list[str]] = {"abstract": [], "decisions": [], "actions": [], "problems": []}
    for section in root:
        tag = section.tag.split("}")[-1]
        if tag in out:
            out[tag] = [" ".join((s.text or "").split()) for s in section if (s.text or "").strip()]
    return out


def load_reference(meeting: str, annot_dir: Path) -> AMIReference:
    words: list[RefWord] = []
    for f in sorted((annot_dir / "words").glob(f"{meeting}.*.words.xml")):
        spk = f.name.split(".")[1]
        words += parse_words_file(f, spk)
    if not words:
        raise FileNotFoundError(f"no word annotations for {meeting} in {annot_dir}")
    ref = AMIReference(meeting=meeting, words=words)
    abs_file = annot_dir / "abstractive" / f"{meeting}.abssumm.xml"
    if abs_file.exists():
        s = parse_abssumm(abs_file)
        ref.abstract, ref.decisions, ref.actions, ref.problems = (
            s["abstract"], s["decisions"], s["actions"], s["problems"])
    return ref


def meetings_in(split: str) -> list[str]:
    if split in SPLITS:
        return SPLITS[split]
    return [m for m in re.split(r"[,\s]+", split) if m]
