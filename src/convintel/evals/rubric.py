"""The summary-quality rubric. Kept in code so the judge prompt, the human
annotation sheet, and docs/rubric.md are generated from one source of truth.

Each dimension has anchored levels: annotators and the judge both score against
the same concrete descriptions, which is what makes calibration meaningful.
"""
from __future__ import annotations

RUBRIC: dict[str, dict] = {
    "faithfulness": {
        "question": "Is everything in the record supported by the transcript?",
        "anchors": {
            5: "Every statement, decision, owner and date is supported by the cited evidence.",
            4: "One minor imprecision (e.g. slightly overstated certainty) but nothing false.",
            3: "One substantive unsupported claim, OR a wrong owner on an action item.",
            2: "Several unsupported claims or a fabricated decision/action.",
            1: "Largely invented or contradicts the transcript.",
        },
    },
    "decision_coverage": {
        "question": "Are the meeting's real decisions captured (vs. the reference decisions)?",
        "anchors": {
            5: "All reference decisions captured; nothing important missing.",
            4: "Most captured; one minor decision missing.",
            3: "About half captured, or one major decision missing.",
            2: "Only a minority captured.",
            1: "None of the reference decisions captured.",
        },
    },
    "action_item_quality": {
        "question": "Are action items real commitments, with correct owners, and none missed?",
        "anchors": {
            5: "All real action items present, owners right, no non-commitments listed as actions.",
            4: "One minor miss or one vague item.",
            3: "Notable miss, or proposals/ideas listed as actions, or one wrong owner.",
            2: "Most items missing or wrong.",
            1: "Action items absent or meaningless.",
        },
    },
    "summary_usefulness": {
        "question": "Could someone who missed the meeting act on this summary?",
        "anchors": {
            5: "Purpose, key discussion, outcomes and next steps are clear and concise.",
            4: "Clear but slightly verbose or misses one useful detail.",
            3: "Gets the gist but omits outcomes or is padded with filler.",
            2: "Vague; reader would still need to ask what happened.",
            1: "Not useful.",
        },
    },
}

DIMENSIONS = list(RUBRIC)


def rubric_text() -> str:
    parts = []
    for name, spec in RUBRIC.items():
        anchors = "\n".join(f"    {k}: {v}" for k, v in sorted(spec["anchors"].items(), reverse=True))
        parts.append(f"- {name}: {spec['question']}\n{anchors}")
    return "\n".join(parts)


def rubric_markdown() -> str:
    out = ["| Dimension | Question | 5 | 3 | 1 |", "|---|---|---|---|---|"]
    for name, spec in RUBRIC.items():
        a = spec["anchors"]
        out.append(f"| {name} | {spec['question']} | {a[5]} | {a[3]} | {a[1]} |")
    return "\n".join(out)
