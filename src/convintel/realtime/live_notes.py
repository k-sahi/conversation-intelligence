"""Incremental meeting record for the real-time path (framework-independent, testable).

Final transcript segments stream in from per-participant STT. Every `update_every_s`
seconds (or `update_every_utts` utterances) we run the SAME map-step extractor used by
the batch pipeline on the new utterances plus a little context, then merge into the
running record. This trades some quality (no global reduce until the meeting ends)
for freshness; `finalize()` runs the full batch reduce over everything.

In real time, diarization comes from track identity: each LiveKit participant publishes
their own mic, so speaker = participant identity and there is no clustering step.
That is the main reason real time can skip pyannote; it fails only when several people
share one room mic (then fall back to Deepgram's streaming `diarize`).
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from ..schemas import ActionItem, ChunkExtraction, Decision, MeetingRecord, Transcript, Utterance
from ..summarize.extractor import MeetingExtractor


def _norm(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def similar(a: str, b: str, thresh: float = 0.6) -> bool:
    x, y = _norm(a), _norm(b)
    return bool(x and y) and len(x & y) / len(x | y) >= thresh


@dataclass
class LiveNotes:
    meeting_id: str
    extractor: MeetingExtractor
    update_every_s: float = 45.0
    update_every_utts: int = 25
    context_utts: int = 6
    utterances: list[Utterance] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    actions: list[ActionItem] = field(default_factory=list)
    window_notes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    _processed: int = 0
    _last_update: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_utterance(self, speaker: str, text: str, start: float, end: float) -> Utterance:
        with self._lock:
            u = Utterance(id=f"u{len(self.utterances):04d}", speaker=speaker, start=start, end=end, text=text.strip())
            self.utterances.append(u)
            return u

    def due(self) -> bool:
        pending = len(self.utterances) - self._processed
        return pending > 0 and (pending >= self.update_every_utts
                                or time.monotonic() - self._last_update >= self.update_every_s)

    def _transcript(self) -> Transcript:
        return Transcript(meeting_id=self.meeting_id, duration=self.utterances[-1].end if self.utterances else 0.0,
                          utterances=list(self.utterances), speakers=sorted({u.speaker for u in self.utterances}),
                          source="realtime")

    def update(self) -> ChunkExtraction | None:
        """Blocking LLM call — run it in a thread executor from async code."""
        with self._lock:
            if self._processed >= len(self.utterances):
                return None
            start = max(0, self._processed - self.context_utts)
            window = self.utterances[start:]
            self._processed = len(self.utterances)
            self._last_update = time.monotonic()
        part = self.extractor.extract_window(self._transcript(), window, idx=len(self.window_notes), total=0)
        self.merge(part)
        return part

    def merge(self, part: ChunkExtraction) -> None:
        with self._lock:
            for d in part.decisions:
                match = next((x for x in self.decisions if similar(x.statement, d.statement)), None)
                if match:
                    match.evidence = sorted(set(match.evidence) | set(d.evidence))
                    match.statement = d.statement  # later wording usually reflects the final state
                else:
                    d.id = f"D{len(self.decisions) + 1}"
                    self.decisions.append(d)
            for a in part.action_items:
                match = next((x for x in self.actions if similar(x.task, a.task)), None)
                if match:
                    match.evidence = sorted(set(match.evidence) | set(a.evidence))
                    match.owner = match.owner or a.owner
                    match.due = match.due or a.due
                else:
                    a.id = f"A{len(self.actions) + 1}"
                    self.actions.append(a)
            for q in part.open_questions:
                if not any(similar(q, x) for x in self.open_questions):
                    self.open_questions.append(q)
            if part.notes:
                self.window_notes.append(part.notes)

    def snapshot(self) -> MeetingRecord:
        with self._lock:
            return MeetingRecord(
                meeting_id=self.meeting_id,
                summary=" ".join(self.window_notes[-4:]) or "Meeting in progress.",
                decisions=[d.model_copy() for d in self.decisions],
                action_items=[a.model_copy() for a in self.actions],
                open_questions=list(self.open_questions),
                model_info={"mode": "realtime-incremental", "utterances": str(len(self.utterances))},
            )

    def recap_text(self, max_items: int = 4) -> str:
        """Short, speakable recap for TTS (no markdown, no ids)."""
        r = self.snapshot()
        parts = []
        if r.decisions:
            parts.append("Decisions so far: " + "; ".join(d.statement for d in r.decisions[-max_items:]) + ".")
        if r.action_items:
            parts.append("Action items: " + "; ".join(
                f"{a.owner + ' will ' if a.owner else ''}{a.task}" for a in r.action_items[-max_items:]) + ".")
        if not parts:
            parts.append("No decisions or action items yet.")
        return " ".join(parts)

    def finalize(self) -> MeetingRecord:
        """End of meeting: full batch extraction (map + reduce + grounding) over the live transcript."""
        return self.extractor.run(self._transcript())
