"""Batch speaker diarization with pyannote `speaker-diarization-community-1`.

Community-1 returns two views of the same audio:

* `speaker_diarization`           – overlapping turns (two people can be active at once).
                                    Used for DER scoring and for flagging overlapped speech.
* `exclusive_speaker_diarization` – at most one speaker at a time (the most likely one).
                                    Used to assign ASR words to speakers, because a word
                                    can only be written down once in the transcript.

Keeping both is the core of our overlap strategy (see docs/design_decisions.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..schemas import SpeakerTurn


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]                       # overlapping view
    exclusive_turns: list[SpeakerTurn]             # non-overlapping view
    model: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})


class PyannoteDiarizer:
    def __init__(self, pipeline_name: str | None = None, token: str | None = None, device: str | None = None):
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:  # pragma: no cover - heavy optional dep
            raise ImportError("pip install 'pyannote.audio>=4.0' torch torchaudio") from exc

        self.name = pipeline_name or settings.pyannote_pipeline
        self.pipeline = Pipeline.from_pretrained(self.name, token=token or settings.hf_token)
        dev = device or settings.pyannote_device
        if dev != "cpu":
            self.pipeline.to(torch.device(dev))

    def __call__(
        self,
        audio_path: str | Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

        # Loading audio into memory first avoids torchcodec/ffmpeg decoding issues
        # on some platforms and is faster for repeated runs.
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(audio_path))
            output = self.pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
        except Exception:  # fall back to letting pyannote decode the file
            output = self.pipeline(str(audio_path), **kwargs)

        turns = _annotation_to_turns(getattr(output, "speaker_diarization", output))
        exclusive_src = getattr(output, "exclusive_speaker_diarization", None)
        exclusive = _annotation_to_turns(exclusive_src) if exclusive_src is not None else make_exclusive(turns)
        return DiarizationResult(turns=turns, exclusive_turns=exclusive, model=self.name, meta=kwargs)


def _annotation_to_turns(annotation) -> list[SpeakerTurn]:
    turns = []
    # Community-1 output iterates as (turn, speaker); legacy Annotation uses itertracks.
    if hasattr(annotation, "itertracks"):
        iterator = ((seg, label) for seg, _, label in annotation.itertracks(yield_label=True))
    else:
        iterator = iter(annotation)
    for seg, label in iterator:
        turns.append(SpeakerTurn(speaker=str(label), start=float(seg.start), end=float(seg.end)))
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def make_exclusive(turns: list[SpeakerTurn], step: float = 0.01) -> list[SpeakerTurn]:
    """Fallback exclusive view: in overlaps, keep the speaker whose turn started last
    (the interrupter is usually the one being transcribed by a single-channel ASR)."""
    if not turns:
        return []
    end = max(t.end for t in turns)
    n = int(end / step) + 1
    owner: list[str | None] = [None] * n
    for t in sorted(turns, key=lambda t: t.start):
        for i in range(int(t.start / step), min(n, int(t.end / step))):
            owner[i] = t.speaker
    out: list[SpeakerTurn] = []
    cur, s = None, 0
    for i, spk in enumerate(owner + [None]):
        if spk != cur:
            if cur is not None:
                out.append(SpeakerTurn(speaker=cur, start=round(s * step, 3), end=round(i * step, 3)))
            cur, s = spk, i
    return out


# ------------------------------------------------------------------- RTTM I/O
def write_rttm(turns: list[SpeakerTurn], uri: str, path: str | Path) -> None:
    lines = [
        f"SPEAKER {uri} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
        for t in turns
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def read_rttm(path: str | Path, uri: str | None = None) -> list[SpeakerTurn]:
    turns = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        if uri and parts[1] != uri:
            continue
        start, dur = float(parts[3]), float(parts[4])
        turns.append(SpeakerTurn(speaker=parts[7], start=start, end=start + dur))
    return sorted(turns, key=lambda t: t.start)
