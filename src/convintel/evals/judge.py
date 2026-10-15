"""LLM-as-judge for a summary that has no single right answer.

Three complementary judges, because no one number captures summary quality:

1. RubricJudge  – scores the 4 rubric dimensions (1-5) against anchored levels.
                  Reasoning is written BEFORE the score (field order in the schema) so
                  the score is conditioned on the analysis, not rationalised after it.
2. ItemMatcher  – atomic, reference-based: for each AMI reference decision / action, is it
                  covered by some system item? -> recall. For each system item, is it
                  supported by the reference? -> (lower-bound) precision. AMI references
                  are sparse, so precision is pessimistic; faithfulness covers the rest.
3. PairwiseJudge – A vs B preference for comparing two system versions, run in BOTH
                  orders to cancel position bias; disagreement between orders = tie.

Hygiene: the judge model differs from the extraction model by default (self-preference
bias), temperature 0, and `n_samples>1` takes the median for stability.
"""
from __future__ import annotations

import json
import statistics
from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import MeetingRecord, Transcript
from ..summarize.llm import LLM
from .rubric import DIMENSIONS, rubric_text

# --------------------------------------------------------------- rubric judge
JUDGE_SYSTEM = """You are an expert evaluator of meeting summaries. You are strict and calibrated:
a 5 is rare and means there is nothing to improve. Judge ONLY against the rubric anchors.
The reference summary was written by a human annotator; it is a guide, not the only valid
answer. A system item that is correct per the transcript but absent from the reference is
NOT an error. Ignore formatting and length unless it hurts usefulness.

Rubric:
{rubric}"""

JUDGE_USER = """<reference_summary>
{reference}
</reference_summary>

<system_record>
{record}
</system_record>

<evidence>
Transcript lines cited by the system (plus neighbours), for checking faithfulness:
{evidence}
</evidence>

For each dimension, first write your analysis, then give an integer score 1-5."""


class DimensionScore(BaseModel):
    dimension: Literal["faithfulness", "decision_coverage", "action_item_quality", "summary_usefulness"]
    analysis: str = Field(..., description="Specific evidence for the score. Written before the score.")
    score: int = Field(..., ge=1, le=5)


class RubricVerdict(BaseModel):
    scores: list[DimensionScore]
    unsupported_claims: list[str] = Field(default_factory=list,
                                          description="Quote any system statements not supported by evidence.")


def _reference_block(abstract: list[str], decisions: list[str], actions: list[str]) -> str:
    return ("ABSTRACT:\n" + " ".join(abstract)
            + "\n\nDECISIONS:\n" + "\n".join(f"- {d}" for d in decisions or ["(none annotated)"])
            + "\n\nACTIONS:\n" + "\n".join(f"- {a}" for a in actions or ["(none annotated)"]))


def _evidence_block(record: MeetingRecord, transcript: Transcript, ctx: int = 2, limit: int = 120) -> str:
    order = {u.id: i for i, u in enumerate(transcript.utterances)}
    keep: set[int] = set()
    for item in [*record.decisions, *record.action_items]:
        for e in item.evidence:
            if e in order:
                keep |= set(range(max(0, order[e] - ctx), order[e] + ctx + 1))
    utts = [transcript.utterances[i] for i in sorted(keep) if i < len(transcript.utterances)][:limit]
    return transcript.as_prompt_text(utts) or "(no evidence cited)"


def _record_block(r: MeetingRecord) -> str:
    return json.dumps({
        "summary": r.summary,
        "decisions": [{"id": d.id, "statement": d.statement, "evidence": d.evidence} for d in r.decisions],
        "action_items": [{"id": a.id, "task": a.task, "owner": a.owner, "due": a.due, "evidence": a.evidence}
                         for a in r.action_items],
        "open_questions": r.open_questions,
    }, indent=1)


class RubricJudge:
    def __init__(self, llm: LLM, n_samples: int = 1):
        self.llm = llm
        self.n_samples = n_samples

    def score(self, record: MeetingRecord, transcript: Transcript, ref_abstract: list[str],
              ref_decisions: list[str], ref_actions: list[str]) -> dict:
        system = JUDGE_SYSTEM.format(rubric=rubric_text())
        user = JUDGE_USER.format(
            reference=_reference_block(ref_abstract, ref_decisions, ref_actions),
            record=_record_block(record),
            evidence=_evidence_block(record, transcript),
        )
        samples = [
            self.llm.structured(system, user, RubricVerdict, tool_name="grade",
                                temperature=0.0 if self.n_samples == 1 else 0.7)
            for _ in range(self.n_samples)
        ]
        out: dict = {"scores": {}, "spread": {}, "analysis": {}, "unsupported_claims": samples[0].unsupported_claims}
        for dim in DIMENSIONS:
            vals = [s.score for v in samples for s in v.scores if s.dimension == dim]
            if vals:
                out["scores"][dim] = float(statistics.median(vals))
                out["spread"][dim] = max(vals) - min(vals)
                out["analysis"][dim] = next(s.analysis for s in samples[0].scores if s.dimension == dim) \
                    if any(s.dimension == dim for s in samples[0].scores) else ""
        out["overall"] = round(statistics.mean(out["scores"].values()), 3) if out["scores"] else None
        return out


# --------------------------------------------------------------- atomic matching
MATCH_SYSTEM = """You compare two lists of meeting items (decisions or action items).
An item MATCHES if it refers to the same outcome/commitment, even if worded differently
or at a different level of detail. Partial matches (same topic, different outcome) do NOT match."""

MATCH_USER = """REFERENCE items:
{ref}

SYSTEM items:
{hyp}

For every reference item, give the index of the matching system item or null.
For every system item, say whether it is supported by ANY reference item."""


class RefMatch(BaseModel):
    ref_index: int
    system_index: int | None


class SysSupport(BaseModel):
    system_index: int
    supported: bool


class MatchVerdict(BaseModel):
    ref_matches: list[RefMatch]
    sys_support: list[SysSupport]


class ItemMatcher:
    def __init__(self, llm: LLM):
        self.llm = llm

    def match(self, ref_items: list[str], sys_items: list[str]) -> dict:
        if not ref_items and not sys_items:
            return {"precision": None, "recall": None, "f1": None, "n_ref": 0, "n_sys": 0}
        if not ref_items or not sys_items:
            return {"precision": 0.0 if sys_items else None, "recall": 0.0 if ref_items else None, "f1": 0.0,
                    "n_ref": len(ref_items), "n_sys": len(sys_items)}
        user = MATCH_USER.format(ref="\n".join(f"{i}. {t}" for i, t in enumerate(ref_items)),
                                 hyp="\n".join(f"{i}. {t}" for i, t in enumerate(sys_items)))
        v = self.llm.structured(MATCH_SYSTEM, user, MatchVerdict, tool_name="match")
        recall = sum(m.system_index is not None for m in v.ref_matches) / len(ref_items)
        precision = sum(s.supported for s in v.sys_support) / len(sys_items)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                "n_ref": len(ref_items), "n_sys": len(sys_items)}


# --------------------------------------------------------------- pairwise
PAIR_SYSTEM = """You compare two meeting records for the same meeting and pick the better one
for a reader who missed the meeting. Prioritise faithfulness, then decisions/actions, then clarity.
Answer A, B, or TIE."""


class PairVerdict(BaseModel):
    reasoning: str
    winner: Literal["A", "B", "TIE"]


class PairwiseJudge:
    def __init__(self, llm: LLM):
        self.llm = llm

    def compare(self, a: MeetingRecord, b: MeetingRecord, reference: str) -> dict:
        def ask(x, y):
            user = f"<reference>\n{reference}\n</reference>\n<A>\n{_record_block(x)}\n</A>\n<B>\n{_record_block(y)}\n</B>"
            return self.llm.structured(PAIR_SYSTEM, user, PairVerdict, tool_name="verdict").winner

        first = ask(a, b)
        second = {"A": "B", "B": "A", "TIE": "TIE"}[ask(b, a)]   # map back to original labels
        consistent = first == second
        return {"winner": first if consistent else "TIE", "order_consistent": consistent,
                "raw": [first, second]}
