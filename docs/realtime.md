# Real-time path

## Latency definitions

All four are measured from the **user's** point of view and reported as p50 / p90 / p95 / max.

| Metric | Start → stop | Why it matters | Where it's measured |
|---|---|---|---|
| **TTFO** time to first output | speech onset (audio clock) → first interim transcript for that segment | captions feel live or laggy | `MultiUserTranscriber`, `bench_realtime.py` |
| **TTFW** time to first word | LLM request → first generated token | the thinking pause | LiveKit `LLMMetrics.ttft`; SSE in bench |
| **WPS** words per second | generated words / (last − first token time) | must stay above speaking rate (~2.5–3 w/s) or TTS stalls mid-sentence | bench (exact); agent (tokens/s × 0.75, estimate) |
| **TTFA** time to first audio | end of user turn → first synthesized audio | the latency people actually feel in a voice exchange | agent: `EOU delay + ttft + TTS ttfb` per speech id; bench: request → first ElevenLabs byte |

Also logged: STT final-transcript lag (segment end → final), notes-update duration, TTS TTFB.

### Why TTFO uses the audio clock
Wall-clock "time since I started streaming" mixes in how long the person has been talking. We anchor each
segment at its onset on the stream's own timeline (`stream_start + segment.start`) so TTFO measures system
delay only.

## Latency budget and levers

```
user stops ─► endpointing (300 ms) ─► LLM TTFW (~0.4-0.7 s) ─► first sentence ─► TTS TTFB (~0.2-0.4 s) ─► audio
               └ turn detector/VAD     └ smaller model, short prompt,      └ stream LLM → TTS by sentence
                 tune vs. false cuts     cached system prompt
```

| Lever | Effect | Trade-off |
|---|---|---|
| Shorter endpointing | lower TTFA | more mid-sentence cut-ins → use LiveKit turn detector instead of silence only |
| Haiku-tier model for answers | lower TTFW, higher WPS | weaker reasoning → answers grounded through the `get_meeting_record` tool |
| Sentence-level streaming LLM → TTS | TTFA ≈ time to first sentence, not full answer | needs a splitter that doesn't break on "e.g." |
| Interim results | TTFO in hundreds of ms | interims get revised; only finals feed the notes |
| Update notes every 45 s, in a thread | no blocking of the audio loop | notes up to ~45 s stale; recap uses the latest snapshot |

## Running the benchmark

```bash
python scripts/bench_realtime.py data/audio/ES2011a.Mix-Headset.wav --seconds 300 --turns 10
```
Streams 5 minutes of AMI audio to Deepgram at 1× speed, then runs 10 recap turns through Claude and
ElevenLabs streaming. Output: `outputs/bench/<name>.json`. In-room numbers (WebRTC, turn detection, network)
come from the agent: `outputs/<room>/realtime_latency.json`.

## Real-time quality

At meeting end `LiveNotes.finalize()` runs the batch extractor over the live transcript, and
`scripts/run_eval.py`-style metrics can be computed on `realtime_transcript.json`. Comparing that record with
the batch record of the same audio quantifies what real time costs in quality.
