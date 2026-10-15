"""Map-reduce extraction of a structured MeetingRecord from a Transcript.

Why map-reduce: an AMI meeting is ~30-40 min, ~5-7k words. It fits in a large
context window, but per-window extraction (a) keeps evidence ids local and accurate,
(b) is what the real-time path does incrementally anyway, so batch and real-time
share one extractor, and (c) parallelises.

After reduce, a grounding pass checks every item's evidence ids exist and that the
cited utterances lexically support it. Ungrounded items are dropped and logged in
`record.warnings` — we prefer a missing item to a hallucinated one.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from ..config import settings
from ..schemas import ActionItem, ChunkExtraction, Decision, MeetingRecord, Topic, Transcript, Utterance
from .llm import LLM
from .prompts import MAP_SYSTEM, MAP_USER, REDUCE_SYSTEM, REDUCE_USER

STOP = set("""a an the and or but if of to in on at for with by from is are was were be been being
it this that these those we you i he she they them our your us me my so just like yeah okay um uh
will would should could can do does did have has had not no yes then than there here what which
who whom how when where all any some about into out up down over again also very really""".split())


class ReduceOutput(BaseModel):
    summary: str
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


def _fmt_t(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"


def chunk_utterances(utts: list[Utterance], max_chars: int, overlap: int) -> list[list[Utterance]]:
    """Greedy windows of at most max_chars, repeating `overlap` utterances between windows
    so a decision that straddles a boundary is seen whole at least once."""
    windows: list[list[Utterance]] = []
    i = 0
    while i < len(utts):
        size, j = 0, i
        while j < len(utts) and (size + len(utts[j].text) + 30 <= max_chars or j == i):
            size += len(utts[j].text) + 30
            j += 1
        windows.append(utts[i:j])
        if j >= len(utts):
            break
        i = max(i + 1, j - overlap)
    return windows


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


def support_score(claim: str, evidence_texts: list[str]) -> float:
    """Fraction of the claim's content words that appear in the cited utterances (+/- 3 lines
    are added by the caller). Cheap lexical proxy; the LLM judge does the semantic check."""
    cw = _content_words(claim)
    if not cw:
        return 1.0
    ev = set().union(*(_content_words(t) for t in evidence_texts)) if evidence_texts else set()
    # loose stemming: match on 5-char prefixes
    evp = {w[:5] for w in ev}
    hits = sum(1 for w in cw if w in ev or w[:5] in evp)
    return hits / len(cw)


class MeetingExtractor:
    def __init__(self, llm: LLM, max_chunk_chars: int | None = None, overlap_utts: int | None = None,
                 min_support: float = 0.15, workers: int = 4):
        self.llm = llm
        self.max_chunk_chars = max_chunk_chars or settings.max_chunk_chars
        self.overlap_utts = settings.chunk_overlap_utts if overlap_utts is None else overlap_utts
        self.min_support = min_support
        self.workers = workers

    # ------------------------------------------------------------------ map
    def extract_window(self, transcript: Transcript, window: list[Utterance], idx: int, total: int) -> ChunkExtraction:
        user = MAP_USER.format(
            meeting_id=transcript.meeting_id, idx=idx + 1, total=total,
            t0=_fmt_t(window[0].start), t1=_fmt_t(window[-1].end),
            transcript=transcript.as_prompt_text(window),
        )
        return self.llm.structured(MAP_SYSTEM, user, ChunkExtraction, tool_name="record_window")

    # ------------------------------------------------------------------ reduce
    def reduce(self, transcript: Transcript, parts: list[ChunkExtraction]) -> ReduceOutput:
        payload = json.dumps([p.model_dump(exclude_none=True) for p in parts], indent=1)
        user = REDUCE_USER.format(
            meeting_id=transcript.meeting_id, duration=_fmt_t(transcript.duration),
            speakers=", ".join(transcript.speakers), extractions=payload,
        )
        return self.llm.structured(REDUCE_SYSTEM, user, ReduceOutput, tool_name="record_meeting", max_tokens=8192)

    # ------------------------------------------------------------------ grounding
    def ground(self, transcript: Transcript, out: ReduceOutput) -> tuple[ReduceOutput, list[str]]:
        by_id = transcript.by_id()
        order = {u.id: i for i, u in enumerate(transcript.utterances)}
        utts = transcript.utterances
        warnings: list[str] = []

        def context(ids: list[str]) -> list[str]:
            texts = []
            for uid in ids:
                i = order[uid]
                texts += [u.text for u in utts[max(0, i - 3): i + 4]]
            return texts

        def check(items, text_of, kind):
            kept = []
            for it in items:
                valid = [e for e in it.evidence if e in by_id]
                if not valid:
                    warnings.append(f"dropped {kind} with no valid evidence: {text_of(it)!r}")
                    continue
                score = support_score(text_of(it), context(valid))
                if score < self.min_support:
                    warnings.append(f"dropped {kind} weakly supported ({score:.2f}): {text_of(it)!r}")
                    continue
                overlapped = sum(by_id[e].overlap_ratio > 0.3 for e in valid) / len(valid)
                conf = it.confidence * (0.85 if overlapped > 0.5 else 1.0)
                kept.append(it.model_copy(update={"evidence": valid, "confidence": round(conf, 3)}))
            return kept

        out = out.model_copy(update={
            "decisions": check(out.decisions, lambda d: d.statement, "decision"),
            "action_items": check(out.action_items, lambda a: a.task, "action"),
        })
        return out, warnings

    # ------------------------------------------------------------------ public
    def run(self, transcript: Transcript) -> MeetingRecord:
        windows = chunk_utterances(transcript.utterances, self.max_chunk_chars, self.overlap_utts)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            parts = list(pool.map(lambda iw: self.extract_window(transcript, iw[1], iw[0], len(windows)),
                                  enumerate(windows)))
        reduced = self.reduce(transcript, parts)
        grounded, warnings = self.ground(transcript, reduced)
        for i, d in enumerate(grounded.decisions):
            d.id = f"D{i + 1}"
        for i, a in enumerate(grounded.action_items):
            a.id = f"A{i + 1}"
        return MeetingRecord(
            meeting_id=transcript.meeting_id,
            summary=grounded.summary,
            topics=grounded.topics,
            decisions=grounded.decisions,
            action_items=grounded.action_items,
            open_questions=grounded.open_questions,
            problems=grounded.problems,
            warnings=warnings,
            model_info={"llm": getattr(self.llm, "model", "?"), "windows": str(len(windows))},
        )
