"""Real-time latency metrics.

Definitions (all in seconds, reported as p50 / p90 / p95 / max over events):

  TTFO  time to first output    speech onset of a segment -> first interim transcript for it.
                                How quickly the system shows *something* for what was said.
  TTFW  time to first word      request to the LLM (end of user turn or recap trigger)
                                -> first generated text token. LiveKit reports this as LLM `ttft`.
  WPS   words per second        generated words / (last token time - first token time).
                                Must exceed speech rate (~2.5-3 w/s) or TTS will starve.
  TTFA  time to first audio     end of user turn -> first synthesized audio frame played.
                                The number a user actually feels; ~= EOU delay + TTFW + TTS TTFB.

We also report the STT final-transcript lag (segment end -> final transcript) because
that is what bounds how fresh the live notes can be.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def summarize(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": round(percentile(xs, 50), 3), "p90": round(percentile(xs, 90), 3),
            "p95": round(percentile(xs, 95), 3), "max": round(max(xs), 3) if xs else float("nan"),
            "mean": round(sum(xs) / len(xs), 3) if xs else float("nan")}


@dataclass
class LatencyTracker:
    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _marks: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    def mark(self, key: str, t: float | None = None) -> None:
        self._marks[key] = self.now() if t is None else t

    def since(self, key: str, metric: str, t: float | None = None, once: bool = True) -> float | None:
        start = self._marks.get(key)
        if start is None:
            return None
        dt = (self.now() if t is None else t) - start
        self.samples[metric].append(dt)
        if once:
            self._marks.pop(key, None)
        return dt

    def add(self, metric: str, value: float) -> None:
        self.samples[metric].append(value)

    def report(self) -> dict:
        return {k: summarize(v) for k, v in self.samples.items()}

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"summary": self.report(), "raw": self.samples}, indent=1))


def words_per_second(token_times: list[float], texts: list[str]) -> float | None:
    if len(token_times) < 2:
        return None
    words = len(" ".join(texts).split())
    dt = token_times[-1] - token_times[0]
    return words / dt if dt > 0 else None
