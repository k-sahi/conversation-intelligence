"""Data model for the whole system.

Two layers:
  * Signal layer  – Word, SpeakerTurn, Utterance, Transcript (what was said, by whom, when)
  * Record layer  – Decision, ActionItem, Topic, MeetingRecord (what it meant)

Every record-layer item carries `evidence`: the utterance ids that support it.
That is what makes the summary auditable and lets us run a grounding check.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------- signal layer
class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: str | None = None
    channel: int | None = None
    overlap: bool = False                   # True if >1 speaker active while this word was spoken


class SpeakerTurn(BaseModel):
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Utterance(BaseModel):
    id: str                                 # stable id, e.g. "u0042" — referenced as evidence
    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    overlap_ratio: float = 0.0              # share of words spoken in overlap
    co_speakers: list[str] = Field(default_factory=list)


class Transcript(BaseModel):
    meeting_id: str
    duration: float
    utterances: list[Utterance]
    speakers: list[str]
    source: Literal["batch", "realtime"] = "batch"
    asr_model: str = ""
    diarization_model: str = ""

    def as_prompt_text(self, utts: list[Utterance] | None = None) -> str:
        """Render as `[u0001 | 00:12 | SPEAKER_00] text` lines for the LLM."""
        rows = []
        for u in utts if utts is not None else self.utterances:
            mm, ss = divmod(int(u.start), 60)
            flag = " (overlap)" if u.overlap_ratio > 0.3 else ""
            rows.append(f"[{u.id} | {mm:02d}:{ss:02d} | {u.speaker}{flag}] {u.text}")
        return "\n".join(rows)

    def by_id(self) -> dict[str, Utterance]:
        return {u.id: u for u in self.utterances}


# ----------------------------------------------------------------- record layer
class Decision(BaseModel):
    id: str = ""
    statement: str = Field(..., description="The decision, phrased as a settled outcome.")
    rationale: str | None = Field(None, description="Why, if stated.")
    decided_by: list[str] = Field(default_factory=list, description="Speaker labels who proposed/agreed.")
    evidence: list[str] = Field(default_factory=list, description="Utterance ids that support this.")
    confidence: float = Field(0.7, ge=0.0, le=1.0)

    @field_validator("statement")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("statement must not be empty")
        return v.strip()


class ActionItem(BaseModel):
    id: str = ""
    task: str
    owner: str | None = Field(None, description="Speaker label or name; null if unassigned.")
    due: str | None = Field(None, description="As stated, e.g. 'next meeting'. Never invented.")
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(0.7, ge=0.0, le=1.0)


class Topic(BaseModel):
    title: str
    start: float | None = None
    end: float | None = None
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)


class SpeakerStats(BaseModel):
    speaker: str
    talk_time_s: float
    turns: int
    words: int
    overlap_time_s: float


class MeetingRecord(BaseModel):
    meeting_id: str
    summary: str = Field(..., description="Abstractive summary, 3-8 sentences.")
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    speaker_stats: list[SpeakerStats] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, description="Pipeline notes, e.g. dropped ungrounded items.")
    model_info: dict[str, str] = Field(default_factory=dict)


# ----------------------------------------------------------------- LLM I/O shapes
class ChunkExtraction(BaseModel):
    """What the map step returns for one transcript window."""
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    notes: str = Field("", description="2-4 sentence summary of this window.")
