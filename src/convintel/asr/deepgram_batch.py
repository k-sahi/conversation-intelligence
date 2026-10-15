"""Deepgram Nova-3 pre-recorded transcription over the REST API.

We call the HTTP endpoint directly instead of the Python SDK: the SDK's
surface changed across major versions (v3 -> v4 -> v5) while `/v1/listen`
has stayed stable, so this keeps the pipeline version-proof.

Modes
-----
* single-channel (SDM / mixed headset): one stream, Deepgram's own `diarize`
  labels are kept as a *fallback* only; pyannote supplies the real speaker turns.
* multichannel (IHM, one headset per speaker): `multichannel=true`, channel
  index == speaker. Cross-talk bleed is removed later in `align.crosstalk_filter`.
"""
from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any

import requests

from ..config import settings
from ..schemas import Word

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Domain vocabulary for AMI's "design a remote control" scenario. Nova-3 key-term
# prompting boosts recall on rare words without fine-tuning.
AMI_KEYTERMS = [
    "remote control", "kinetic", "scroll wheel", "LCD", "titanium", "rubber",
    "Real Reaction", "fruit and vegetables", "spongy", "prototype", "industrial designer",
    "user interface", "marketing expert", "project manager", "joystick", "push-button",
]


class DeepgramError(RuntimeError):
    pass


class DeepgramBatchASR:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 600.0):
        self.api_key = api_key or settings.deepgram_api_key
        self.model = model or settings.deepgram_model
        self.timeout = timeout
        if not self.api_key:
            raise DeepgramError("DEEPGRAM_API_KEY is not set")

    def _params(self, multichannel: bool, keyterms: list[str] | None, language: str) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = [
            ("model", self.model),
            ("language", language),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("utterances", "true"),
            ("filler_words", "true"),   # AMI references keep "um"/"uh"; normalizer removes them symmetrically
            ("diarize", "false" if multichannel else "true"),
        ]
        if multichannel:
            params.append(("multichannel", "true"))
        for term in keyterms or []:
            params.append(("keyterm", term))
        return params

    def transcribe_file(
        self,
        path: str | Path,
        multichannel: bool = False,
        keyterms: list[str] | None = None,
        language: str = "en",
        retries: int = 3,
    ) -> dict[str, Any]:
        """Return the raw Deepgram JSON response."""
        path = Path(path)
        ctype = mimetypes.guess_type(path.name)[0] or "audio/wav"
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": ctype}
        params = self._params(multichannel, keyterms, language)
        last: Exception | None = None
        for attempt in range(retries):
            try:
                with path.open("rb") as fh:
                    r = requests.post(DEEPGRAM_URL, params=params, headers=headers, data=fh, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise DeepgramError(f"retryable status {r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    raise DeepgramError(f"status {r.status_code}: {r.text[:500]}")
                return r.json()
            except (requests.RequestException, DeepgramError) as exc:
                last = exc
                if isinstance(exc, DeepgramError) and "retryable" not in str(exc):
                    raise
                time.sleep(2 ** attempt)
        raise DeepgramError(f"Deepgram failed after {retries} attempts: {last}")


def parse_deepgram_response(resp: dict[str, Any], multichannel: bool = False) -> list[Word]:
    """Flatten a Deepgram response into Word objects.

    single-channel: speaker = "dg_<n>" from Deepgram's own diarizer (fallback only)
    multichannel:   speaker = "ch<i>", channel = i
    """
    words: list[Word] = []
    channels = resp.get("results", {}).get("channels", [])
    for ch_idx, channel in enumerate(channels):
        alts = channel.get("alternatives") or [{}]
        for w in alts[0].get("words", []):
            text = w.get("punctuated_word") or w.get("word", "")
            if multichannel:
                spk, chan = f"ch{ch_idx}", ch_idx
            else:
                spk = f"dg_{w['speaker']}" if "speaker" in w else None
                chan = None
            words.append(
                Word(
                    text=text,
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=float(w.get("confidence", 1.0)),
                    speaker=spk,
                    channel=chan,
                )
            )
    words.sort(key=lambda x: (x.start, x.end))
    return words
