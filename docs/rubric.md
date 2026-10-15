# Summary-quality rubric

Generated from `src/convintel/evals/rubric.py` — edit there, not here. The same text is
given to human annotators and to the LLM judge.

General rules: score each dimension independently; a 5 means nothing to improve; a correct item that is
missing from the AMI reference is **not** an error; ignore formatting unless it hurts usefulness.

## faithfulness
*Is everything in the record supported by the transcript?*

| Score | Anchor |
|---|---|
| 5 | Every statement, decision, owner and date is supported by the cited evidence. |
| 4 | One minor imprecision (e.g. slightly overstated certainty) but nothing false. |
| 3 | One substantive unsupported claim, OR a wrong owner on an action item. |
| 2 | Several unsupported claims or a fabricated decision/action. |
| 1 | Largely invented or contradicts the transcript. |

## decision_coverage
*Are the meeting's real decisions captured (vs. the reference decisions)?*

| Score | Anchor |
|---|---|
| 5 | All reference decisions captured; nothing important missing. |
| 4 | Most captured; one minor decision missing. |
| 3 | About half captured, or one major decision missing. |
| 2 | Only a minority captured. |
| 1 | None of the reference decisions captured. |

## action_item_quality
*Are action items real commitments, with correct owners, and none missed?*

| Score | Anchor |
|---|---|
| 5 | All real action items present, owners right, no non-commitments listed as actions. |
| 4 | One minor miss or one vague item. |
| 3 | Notable miss, or proposals/ideas listed as actions, or one wrong owner. |
| 2 | Most items missing or wrong. |
| 1 | Action items absent or meaningless. |

## summary_usefulness
*Could someone who missed the meeting act on this summary?*

| Score | Anchor |
|---|---|
| 5 | Purpose, key discussion, outcomes and next steps are clear and concise. |
| 4 | Clear but slightly verbose or misses one useful detail. |
| 3 | Gets the gist but omits outcomes or is padded with filler. |
| 2 | Vague; reader would still need to ask what happened. |
| 1 | Not useful. |

## Annotation workflow

1. Read the AMI reference abstract/decisions/actions, then the system record.
2. For faithfulness, open the cited utterance ids in the transcript; do not rely on memory of the meeting.
3. Score, then write one sentence of justification per dimension in `notes`.
4. Two annotators label every item independently; disagreements of 2+ points are discussed and the rubric
   anchor is clarified if the cause was ambiguity (re-label affected items afterwards).
