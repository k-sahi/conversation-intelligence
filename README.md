# Conversation Intelligence System

Takes a recorded or live conversation and produces a **structured meeting record**: a summary,
decisions, action items (with owners), open questions, and per-speaker statistics, where every
item cites the transcript lines that support it.

Built and evaluated on the **AMI Meeting Corpus** (~100 h of real meetings with human
transcripts, speaker labels, and written summaries).

```
                    ┌──────────────── BATCH (recordings) ────────────────┐
 audio file ──► Deepgram Nova-3 ──► words+times ─┐                       │
            └─► pyannote community-1 ──► turns ──┴► align ──► utterances ─┼─► map-reduce ─► grounding ─► MeetingRecord
                    (overlapping + exclusive)      overlap flags          │   extractor      check       (JSON / MD)
                                                                          │   (Claude)
                    ┌──────────────── REAL TIME (LiveKit room) ──────────┐│
 participant mics ─► Deepgram streaming (1 stream / participant) ─► live notes (same extractor, incremental)
                     speaker = participant identity                     │ └► room text topic "meeting.notes"
 "Notes, recap" ──► Claude ──► ElevenLabs Eleven v3 Conversational ──► spoken answer
                                                                          
 Evals: WER · cpWER · DER (strict / collar / no-overlap) · LLM judge (calibrated rubric) · item P/R
        TTFO · TTFW · WPS · TTFA       Traces + eval scores → Arize
```

## What this project demonstrates

| Question a hiring manager asks | Where it's answered |
|---|---|
| Why these models? | [docs/design_decisions.md §1](docs/design_decisions.md#1-model-choice) |
| How did you handle overlapping speakers? | [§2](docs/design_decisions.md#2-overlapping-speech) + `align.py`, `diarization/` |
| How do you evaluate a summary with no single right answer? | [§3](docs/design_decisions.md#3-evaluating-a-summary-with-no-single-right-answer), [docs/evaluation.md](docs/evaluation.md), `evals/judge.py`, `evals/calibration.py` |
| Batch vs. real time? | [§4](docs/design_decisions.md#4-batch-vs-real-time), [docs/realtime.md](docs/realtime.md) |

## Quick start

```bash
# 0. No keys needed: unit tests + an end-to-end demo on a synthetic AMI-format meeting
python -m unittest discover -s tests        # 41 tests
python scripts/demo_offline.py              # transcript, record, WER/cpWER/DER

# 1. Real run
cp .env.example .env                        # add DEEPGRAM_API_KEY, HF_TOKEN, ANTHROPIC_API_KEY, ...
pip install -r requirements.txt && pip install -e . --no-deps
python scripts/download_ami.py --split dev-mini
python scripts/run_batch.py data/audio/ES2011a.Mix-Headset.wav --keyterms ami

# 2. Evaluate on AMI
python scripts/run_eval.py --split dev-mini                 # everything incl. LLM judge
python scripts/run_eval.py --split test --no-summary        # WER / cpWER / DER only

# 3. API
uvicorn convintel.api.main:app --port 8000
curl -F file=@meeting.wav localhost:8000/v1/meetings        # -> {"id": "...", "status": "queued"}

# 4. Real-time agent (LiveKit)
pip install -r requirements-realtime.txt
python -m convintel.realtime.agent dev                      # then join the room from LiveKit's Agent Console
python scripts/bench_realtime.py data/audio/ES2011a.Mix-Headset.wav   # latency benchmark
```

The pyannote model is gated: accept its terms at
`huggingface.co/pyannote/speaker-diarization-community-1` before the first run.

## Repository map

```
src/convintel/
  config.py              settings from env / .env
  schemas.py             Word, SpeakerTurn, Utterance, Transcript, Decision, ActionItem, MeetingRecord
  asr/deepgram_batch.py  Nova-3 REST client (single- and multi-channel), response parser
  diarization/           pyannote community-1 wrapper, RTTM I/O, exclusive-view fallback
  align.py               word→speaker assignment, overlap flags, utterances, cross-talk filter, stats
  summarize/             LLM client (forced tool use → Pydantic), prompts, map-reduce extractor,
                         grounding check, heuristic offline baseline
  pipeline.py            batch orchestrator with per-stage caching
  realtime/              live_notes.py (framework-free incremental record), agent.py (LiveKit worker)
  api/main.py            FastAPI service
  evals/                 wer.py (WER, cpWER), der.py, judge.py, rubric.py, calibration.py,
                         latency.py, text_metrics.py, runner.py
  observability.py       Arize / OpenTelemetry spans (no-op without keys)
  data/ami.py            AMI download, NXT parsing, standard splits, reference builders
scripts/                 download_ami, run_batch, run_eval, calibrate_judge, bench_realtime, demo_offline
tests/                   41 unit tests + synthetic AMI-format fixture
docs/                    architecture, design decisions, evaluation, rubric, real time, deployment, API, data,
                         Question_Bank.pdf (95 interview Q&As; regenerate: python docs/question_bank/build_pdf.py)
Dockerfile.api / Dockerfile.agent / docker-compose.yml / livekit.toml
```

## Results

Fill this table from `outputs/eval/<run>/aggregate.json` after running the evals; the
numbers below are deliberately left blank rather than guessed. The pyannote model card
reports community-1 DER of 17.0 % (AMI IHM) and 19.9 % (AMI SDM) with no collar and overlap
scored, which is the reference point for the diarization row.

| Condition | WER | cpWER | DER strict | DER collar 0.25 | Decision recall | Faithfulness (1-5) |
|---|---|---|---|---|---|---|
| AMI test, Mix-Headset | | | | | | |
| AMI test, Array1-01 (SDM) | | | | | | |
| AMI test, SDM, oracle #speakers | | | | | | |

| Real time (p50 / p95) | TTFO | TTFW | WPS | TTFA |
|---|---|---|---|---|
| bench_realtime.py, ES2011a | | | | |

## Status and honest limits

* Everything except vendor calls is unit-tested here. The Deepgram, pyannote, Anthropic,
  ElevenLabs and LiveKit integrations follow each vendor's current docs but need your keys
  to run; `agent.py` targets `livekit-agents ~=1.8` and that API changes frequently.
* The LLM judge must be calibrated against human labels before its scores mean anything;
  `calibration/` ships the protocol and template, not fabricated labels.
* AMI is English, scenario-driven (a design team building a remote control), and recorded
  in rooms; expect different numbers on phone calls or open-domain meetings.

## License

Code: MIT. AMI data: CC BY 4.0 (cite Carletta et al., 2005). pyannote community-1: CC BY 4.0.
