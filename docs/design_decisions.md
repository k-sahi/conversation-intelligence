# Design decisions

The four decisions a reviewer will probe, with the reasoning, the alternatives, and how each is
measured. Numbers marked "target" are goals to verify with the eval scripts, not results.

---

## 1. Model choice

### ASR: Deepgram Nova-3 (batch REST + streaming WebSocket)

| Need | Why Nova-3 |
|---|---|
| One model for batch **and** real time | Same model family behind `/v1/listen` (pre-recorded) and the streaming socket, so batch-vs-real-time comparisons isolate the *mode*, not the model. |
| Meeting vocabulary | Key-term prompting (`keyterm=`) boosts rare domain words without fine-tuning; AMI's scenario terms ("kinetic", "scroll wheel", "LCD") are passed by default. |
| Word timestamps + confidences | Required for aligning words to diarization turns. |
| Multichannel | `multichannel=true` transcribes each headset separately (IHM mode). |
| Ops | Managed, scales without GPUs; we call the REST API directly so SDK major-version churn can't break the pipeline. |

**Alternatives considered.** Whisper-large-v3 / distil-whisper self-hosted: strong WER, free per minute,
but no native streaming, hallucinations on silence, and GPU ops. WhisperX adds alignment + diarization but
inherits the same streaming gap. AssemblyAI / Speechmatics: comparable managed options; the choice is
replaceable because `asr/` is an adapter returning `list[Word]`.
**How it's measured:** WER and cpWER on the AMI test split, Mix-Headset and SDM.

### Diarization: pyannote `speaker-diarization-community-1` (batch only)

* Open weights (CC BY 4.0), runs on-prem: audio does not have to leave our infrastructure for this step.
* Better than the previous open pipeline (3.1) on every AMI condition in its model card
  (AMI IHM 18.8 → 17.0 % DER, AMI SDM 22.7 → 19.9 %, no collar, overlap scored).
* Returns **both** an overlapping and an *exclusive* diarization — the exclusive view is exactly what
  word-to-speaker assignment needs (see §2).
* Accepts `num_speakers` or `min/max_speakers`. We run *blind* (2–8) by default and report oracle-count
  runs separately, because oracle counts are not available in production.

**Why not Deepgram's built-in `diarize`?** It is convenient and we keep it as a fallback label, but it is a
black box we can't tune, can't run on-prem, and doesn't expose overlap. **Why not pyannote precision-2?**
Better DER (AMI SDM 15.6 % per the card) but hosted and paid; it is a one-line swap
(`PYANNOTE_PIPELINE`) if accuracy beats cost.

### LLM: Claude, with three tiers

| Role | Default | Why |
|---|---|---|
| Extraction (map/reduce) | `claude-sonnet-5-5` | Good structured-output reliability and long-context reading at moderate cost. |
| Judge | `claude-opus-5-5` | Strongest reasoning, and **different from the extractor** to limit self-preference bias. |
| Real-time answers | `claude-haiku-5-5` | Lowest time-to-first-token; answers are short and grounded via a tool. |

Structured output uses **forced tool use**: the tool's `input_schema` is the Pydantic schema, so the
model's tool input *is* the object. Validation failures get one repair round.

### TTS: ElevenLabs Eleven v3 Conversational

The expressive v3 model variant built for real-time agents. It is used only for spoken recaps and answers,
so its latency is on the critical path for TTFA but not for transcription. ElevenLabs is no longer served
through LiveKit Inference, so the agent uses the ElevenLabs plugin with our own key; if an older plugin
version rejects the model id, `ELEVEN_MODEL=eleven_flash_v2_5` is the low-latency fallback.

### Framework: LiveKit Agents on LiveKit Cloud

WebRTC transport, per-participant tracks (which give us speaker identity for free), turn detection, VAD,
interruption handling, and a metrics event stream (`metrics_collected`) that maps directly onto our latency
definitions. Deployed as a Docker image with `lk agent create`.

---

## 2. Overlapping speech

A sizeable share of AMI speech time is overlapped (the `overlap_share` field in every DER result measures it per meeting), and overlap clusters around exactly the moments that matter:
agreement ("yeah, let's do that"), interruptions, and decisions. A pipeline that ignores it loses both
words and attributions at the worst possible time.

**What goes wrong without handling it**
1. Single-channel ASR transcribes only the dominant speaker → the other speaker's words are *missed* (DER miss, WER deletions).
2. Diarization that assigns one speaker per frame is charged a miss for the second speaker (DER).
3. Words get attributed to whoever's turn they fall in → *who said what* breaks even when WER is fine.
   This is invisible to WER and is why we report **cpWER** (fixture demo: WER 3 % vs cpWER 12 %).

**What we do — layered**

| Layer | Technique | Code |
|---|---|---|
| Diarization | Keep pyannote's **overlapping** view for scoring and overlap detection; use the **exclusive** view for assigning each word to exactly one speaker. | `diarization/pyannote_diar.py` |
| Alignment | Word → exclusive turn with maximum temporal overlap; words in gaps snap to the nearest turn ≤0.5 s. Each word gets `overlap=True` when 2+ speakers are active; utterances carry `overlap_ratio` and `co_speakers`. | `align.assign_speakers`, `build_utterances` |
| Summarization | Overlapped lines are marked `(overlap)` in the prompt and the model is told attribution there is less reliable; grounding multiplies confidence by 0.85 for items whose evidence is mostly overlapped. | `summarize/prompts.py`, `extractor.ground` |
| Hardware (when available) | Per-speaker channels: IHM headsets (`--multichannel`) or per-participant tracks in LiveKit remove the separation problem at the source; a 6 dB energy rule removes headset bleed. | `align.crosstalk_filter`, `realtime/agent.py` |
| Measurement | DER reported **strict**, **collar 0.25 s**, and **skip-overlap**; the gap between strict and skip-overlap is the cost of overlap. cpWER vs WER is the attribution cost. `speaker_stats.overlap_time_s` per speaker. | `evals/der.py`, `evals/wer.py` |

**What we deliberately don't do (yet).** Speech separation (e.g. a separation model before ASR) can recover
the hidden speaker's words on single-mic audio, but it adds latency, GPU cost and artefacts that can raise
WER on non-overlapped speech. It is the next experiment if SDM cpWER is the bottleneck, gated on the
skip-overlap vs strict DER gap being large.

---

## 3. Evaluating a summary with no single right answer

Ten good summaries of one meeting can share few words, so n-gram overlap with AMI's single human abstract
(ROUGE) is a weak signal. We decompose "is this record good?" into parts that *can* be checked, and use
a calibrated judge only where semantics are required.

| Layer | Metric | Reference needed? | What it catches |
|---|---|---|---|
| 1. Structural | Pydantic schema validity, evidence ids exist | No | malformed output, invented ids |
| 2. Grounding | lexical support of each item by its cited lines (+/-3 lines) | No | hallucinated decisions/actions (dropped, logged) |
| 3. Atomic recall/precision | LLM matcher: each AMI reference decision/action → covered by a system item? | AMI `abssumm` decisions/actions | missed decisions; noise |
| 4. Rubric judge | 4 dimensions × 1–5 anchored scale: faithfulness, decision coverage, action-item quality, usefulness | AMI abstract as a guide + cited transcript lines | everything semantic |
| 5. Pairwise | A vs B in both orders; inconsistent → tie | No | regressions between versions |
| 6. Sanity | ROUGE-L, compression ratio | AMI abstract | collapses only; never used to rank |

**Making the judge trustworthy**
* **Anchored rubric**: every score level has a concrete description (`evals/rubric.py`, `docs/rubric.md`),
  shared verbatim by annotators and the judge.
* **Analysis before score** (schema field order), **temperature 0**, optional `n_samples=3` median with spread reported.
* **Evidence in context**: the judge sees the cited transcript lines, so faithfulness is checked, not guessed.
* **Different model family/tier from the extractor**, to reduce self-preference.
* **Position-bias control** in pairwise mode (swap order).
* **Calibration** against humans (`evals/calibration.py`): ~30 records, 2 annotators. Inter-annotator QWK is
  the ceiling; the judge must reach **QWK ≥ 0.6 and |mean bias| ≤ 0.5** on every dimension on a held-out half.
  Rubric wording and few-shot anchors are tuned only on the dev half.
* **Known-bad control**: the rule-based `HeuristicLLM` baseline must score clearly lower; if the judge
  can't separate it from the real extractor, the judge is broken, not the extractor.

---

## 4. Batch vs. real time

They are different products with different constraints, sharing one extractor and one schema.

| | Batch | Real time |
|---|---|---|
| Input | finished recording | live WebRTC tracks |
| Speaker attribution | pyannote clustering over the whole meeting (global context → better speaker counting) | track identity (one participant = one speaker); Deepgram streaming diarize only for shared mics |
| ASR | Nova-3 pre-recorded: full-context formatting | Nova-3 streaming: interim results for low TTFO, finals after endpointing |
| Summary | map → **global reduce** → grounding | incremental map every 45 s, merge by similarity; final batch reduce at meeting end |
| Primary metrics | WER, cpWER, DER, judge scores | TTFO, TTFW, WPS, TTFA (+ same quality metrics on the final record) |
| Cost driver | GPU-minutes (pyannote), LLM tokens once | always-on STT per participant, repeated LLM updates |
| Failure mode | slow but complete | fresh but partial; early decisions can be revised later |

**Decision rule we use:** default to **batch** for post-meeting records — it is cheaper, more accurate
(global diarization, global reduce), and the user isn't waiting live. Use **real time** only when the value
is *during* the meeting: live captions, "what did we just decide?", or catching an action item before people
leave. Even then, the real-time record is replaced by a batch pass at the end, so the stored artefact always
has batch quality.

**Latency budget (targets to validate with `bench_realtime.py`)**

| Stage | Target |
|---|---|
| TTFO (onset → first interim) | < 500 ms p50 |
| Endpointing (silence before end-of-turn) | 300 ms configured |
| TTFW (LLM first token) | < 700 ms p50 |
| WPS (LLM words/s) | > 3× speech rate (≈ 8 w/s) so TTS never starves |
| TTS first byte | < 400 ms |
| **TTFA** (end of turn → first audio) | **< 1.5 s p50, < 2.5 s p95** |
