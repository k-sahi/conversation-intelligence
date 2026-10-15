"""A rule-based stand-in for the LLM.

Two uses:
  1. Offline demo / tests: exercises the full map-reduce-ground pipeline with no API key.
  2. A deliberately weak *baseline*: an LLM judge that cannot tell this apart from the
     real extractor is not a judge worth trusting (a sanity check during calibration).
"""
from __future__ import annotations

import json
import re

from .llm import FakeLLM

LINE = re.compile(r"\[(u\d+) \| [\d:]+ \| ([^\]\(]+?)(?: \(overlap\))?\] (.*)")
DECISION_CUES = ("we go with", "we'll go with", "let's go with", "we decided", "agreed", "we will use", "settled on")
ACTION_CUES = ("i will", "i'll", "you will", "can you", "please send", "will prepare", "will send")
QUESTION_CUES = ("what about", "how about", "should we")


def _map(user: str) -> dict:
    decisions, actions, questions, notes = [], [], [], []
    for m in LINE.finditer(user):
        uid, spk, text = m.group(1), m.group(2).strip(), m.group(3).strip()
        low = text.lower()
        if any(c in low for c in DECISION_CUES):
            decisions.append({"statement": text.rstrip(".") + ".", "decided_by": [spk], "evidence": [uid], "confidence": 0.6})
        elif any(c in low for c in ACTION_CUES):
            due = next((d for d in ("friday", "monday", "tomorrow", "next meeting", "next week") if d in low), None)
            actions.append({"task": text.rstrip(".") + ".", "owner": spk, "due": due, "evidence": [uid], "confidence": 0.6})
        elif any(low.startswith(c) for c in QUESTION_CUES) or low.endswith("?"):
            questions.append(text)
        notes.append(text)
    return {"decisions": decisions, "action_items": actions, "topics": [], "open_questions": questions,
            "problems": [], "notes": " ".join(notes[:3])[:400]}


def _reduce(user: str) -> dict:
    blob = user.split("Per-window extractions (JSON):", 1)[1].split("Produce the final", 1)[0]
    parts = json.loads(blob)
    out = {"summary": "", "topics": [], "decisions": [], "action_items": [], "open_questions": [], "problems": []}
    seen = set()
    for p in parts:
        for key, field in (("decisions", "statement"), ("action_items", "task")):
            for it in p.get(key, []):
                if it[field] not in seen:
                    seen.add(it[field])
                    out[key].append(it)
        out["open_questions"] += [q for q in p.get("open_questions", []) if q not in out["open_questions"]]
    out["summary"] = " ".join(p.get("notes", "") for p in parts)[:800] or "No content."
    return out


def _handler(system: str, user: str, schema):
    name = getattr(schema, "__name__", "")
    if name == "ChunkExtraction":
        return _map(user)
    if name == "ReduceOutput":
        return _reduce(user)
    raise ValueError(f"HeuristicLLM cannot produce {name}")


def HeuristicLLM() -> FakeLLM:  # noqa: N802 - reads like a class at call sites
    return FakeLLM(_handler, model="heuristic-baseline")
