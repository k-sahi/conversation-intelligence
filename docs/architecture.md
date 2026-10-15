# Architecture

## 1. Data model (two layers)

**Signal layer** — what was said, by whom, when.

| Type | Fields | Notes |
|---|---|---|
| `Word` | text, start, end, confidence, speaker, channel, overlap | from ASR; `overlap=True` if 2+ speakers active |
| `SpeakerTurn` | speaker, start, end | from diarization (two views: overlapping and exclusive) |
| `Utterance` | id `u0042`, speaker, start, end, text, words, overlap_ratio, co_speakers | consecutive same-speaker words, gap < 0.8 s, max 30 s |
| `Transcript` | meeting_id, duration, utterances, speakers, source | rendered to the LLM as `[u0042 \| 03:15 \| SPEAKER_B (overlap)] text` |

**Record layer** — what it meant.

| Type | Key fields |
|---|---|
| `Decision` | statement, rationale, decided_by, **evidence** (utterance ids), confidence |
| `ActionItem` | task, owner, due (never invented), **evidence**, confidence |
| `Topic`, `SpeakerStats` | chronology, talk time, turns, overlap time |
| `MeetingRecord` | summary, topics, decisions, action_items, open_questions, problems, speaker_stats, warnings |

Evidence ids are the backbone: they make every item auditable in a UI (click → jump to audio),
enable the automatic grounding check, and let the judge verify faithfulness against the cited lines.

## 2. Batch pipeline (`pipeline.py`)

```
audio ─► [ASR] Deepgram Nova-3 REST ─────────────► deepgram.json (cached)
      └► [DIAR] pyannote community-1 ────────────► diarization.rttm + diarization_exclusive.rttm (cached)
             ▼
      [ALIGN] word → exclusive turn with max overlap; snap ≤0.5 s; overlap flag from overlapping view
             ▼
      [UTTER] group words → utterances; relabel SPEAKER_A.. in order of appearance  → transcript.json
             ▼
      [MAP]   per ~12k-char window (6-utterance overlap), Claude via forced tool use → ChunkExtraction
             ▼                                                          (parallel, 4 workers)
      [REDUCE] merge/dedupe, resolve reversals, write summary → ReduceOutput
             ▼
      [GROUND] drop items with no valid evidence or lexical support < 0.15; down-weight overlap items
             ▼
      record.json + record.md + timings.json
```

Each stage is cached under `outputs/<meeting_id>/`, so changing the prompt re-runs only the LLM stages,
and changing the rubric re-runs only the judge.

**Multichannel (IHM) variant.** With one headset per speaker, `multichannel=true` makes the channel
index the speaker; pyannote is skipped and `crosstalk_filter` removes words a headset picked up from a
neighbour (kept only if that channel's energy is within 6 dB of the loudest channel during the word).

## 3. Real-time path (`realtime/`)

```
LiveKit room
  ├─ participant mics ──► MultiUserTranscriber: one Deepgram streaming STT per audio track
  │                        speaker = participant identity (no clustering needed)
  │                        final segments ──► LiveNotes.add_utterance()
  │                        every 45 s or 25 utterances ──► LiveNotes.update() (map step, thread pool)
  │                                                   └──► publish record on text topic "meeting.notes"
  ├─ AgentSession (Deepgram STT → Claude Haiku → ElevenLabs v3 Conversational, Silero VAD)
  │     tool get_meeting_record() lets the agent answer "what did we decide about X?"
  │     "recap" keyword → session.say(LiveNotes.recap_text())
  └─ on shutdown ──► LiveNotes.finalize() = full batch map-reduce-ground over the live transcript
                     + realtime_latency.json (TTFO/TTFW/WPS/TTFA)
```

`LiveNotes` has no LiveKit dependency and is unit-tested; `agent.py` is the thin adapter.

## 4. Service (`api/main.py`)

FastAPI with background jobs and a JSON-on-disk job store. Upload → 202 + id → poll → fetch transcript /
record (JSON or Markdown) → optional ElevenLabs recap audio. `/v1/livekit/token` mints a room token that
also dispatches the `meeting-notes` agent into the room.

Scaling path: the job store and `BackgroundTasks` are behind small interfaces; production swaps them for
Postgres/Redis and a queue (Arq/Celery) with GPU workers for pyannote.

## 5. Observability (`observability.py`)

OpenTelemetry spans exported to Arize with OpenInference attributes. One trace per meeting:
`pipeline → asr.deepgram / diarization.pyannote / align / extract → eval`. Eval scores are set as
`eval.*` span attributes so traces can be filtered by e.g. low faithfulness, and regressions in
WER/DER/judge scores are visible per model version. Without keys every span is a no-op.

## 6. Failure handling

| Failure | Behaviour |
|---|---|
| Deepgram 429/5xx | exponential backoff, 3 attempts; 4xx fails fast with body |
| Anthropic 429/5xx/529 | backoff up to 4 attempts |
| LLM output fails schema | one repair turn with the validation error as `tool_result is_error` |
| Hallucinated / unsupported item | dropped by grounding, reason logged in `record.warnings` |
| Word outside any diarization turn | snapped to nearest turn ≤0.5 s, else ASR label, else UNKNOWN |
| pyannote decode error | falls back from in-memory waveform to file path decoding |
| Job crash | status `failed` with error message, artifacts from completed stages kept |
