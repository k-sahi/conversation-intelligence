"""Component-level real-time latency benchmark on AMI audio (no LiveKit room required).

Streams a WAV file to Deepgram's live endpoint at 1x speed (as a microphone would),
then simulates "Notes, recap" turns: Claude streams a recap and, as soon as the first
sentence is complete, it is sent to ElevenLabs streaming TTS.

Measures:
  TTFO  speech onset (audio clock)  -> first interim transcript for that segment
  STT final lag  segment end        -> final transcript
  TTFW  LLM request                 -> first text token
  WPS   LLM words / streaming duration
  TTFA  LLM request (= end of turn) -> first TTS audio byte   (+ endpointing delay, reported separately)

  python scripts/bench_realtime.py data/audio/ES2011a.Mix-Headset.wav --seconds 300 --turns 10
Results -> outputs/bench/<name>.json

For end-to-end numbers inside LiveKit (network + WebRTC + turn detection), run the agent
and read outputs/<room>/realtime_latency.json, which uses LiveKit's own metrics events.
"""
import argparse
import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convintel.config import settings  # noqa: E402
from convintel.evals.latency import LatencyTracker, words_per_second  # noqa: E402
from convintel.summarize.llm import AnthropicLLM  # noqa: E402

ENDPOINTING_MS = 300


async def stream_stt(path: Path, seconds: float, tracker: LatencyTracker) -> list[str]:
    import websockets  # pip install websockets

    with wave.open(str(path), "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        assert sw == 2, "expects 16-bit PCM"
        frames_per_chunk = int(sr * 0.1)
        url = (f"wss://api.deepgram.com/v1/listen?model={settings.deepgram_model}&encoding=linear16"
               f"&sample_rate={sr}&channels={ch}&interim_results=true&smart_format=true"
               f"&endpointing={ENDPOINTING_MS}&diarize=true")
        finals: list[str] = []
        async with websockets.connect(url, additional_headers={"Authorization": f"Token {settings.deepgram_api_key}"}) as ws:
            t0 = time.perf_counter()
            seen: set[float] = set()

            async def sender():
                sent = 0.0
                while sent < seconds:
                    data = wf.readframes(frames_per_chunk)
                    if not data:
                        break
                    await ws.send(data)
                    sent += 0.1
                    # pace at real time against the wall clock (no drift accumulation)
                    await asyncio.sleep(max(0.0, t0 + sent - time.perf_counter()))
                await ws.send(json.dumps({"type": "CloseStream"}))

            async def receiver():
                async for msg in ws:
                    m = json.loads(msg)
                    if m.get("type") != "Results":
                        continue
                    alt = m["channel"]["alternatives"][0]
                    if not alt.get("transcript"):
                        continue
                    now = time.perf_counter()
                    start, dur = m["start"], m["duration"]
                    if not m["is_final"]:
                        key = round(start, 1)
                        if key not in seen:
                            seen.add(key)
                            first_word = alt["words"][0]["start"] if alt.get("words") else start
                            tracker.add("ttfo_s", now - (t0 + first_word))
                    else:
                        tracker.add("stt_final_lag_s", now - (t0 + start + dur))
                        finals.append(alt["transcript"])

            await asyncio.gather(sender(), receiver())
    return finals


def llm_tts_turn(llm: AnthropicLLM, context: str, tracker: LatencyTracker) -> None:
    t_req = time.perf_counter()
    times, texts, buf, tts_started = [], [], "", False
    for delta in llm.stream_text("You are a meeting note-taker. Give a spoken recap in 2-3 short sentences, "
                                 "no lists or symbols.", f"Transcript so far:\n{context}\n\nNotes, recap please.",
                                 max_tokens=200):
        now = time.perf_counter()
        if not times:
            tracker.add("ttfw_s", now - t_req)
        times.append(now)
        texts.append(delta)
        buf += delta
        if not tts_started and re.search(r"[.!?]\s", buf + " "):
            tts_started = True
            first_sentence = re.split(r"(?<=[.!?])\s", buf, maxsplit=1)[0]
            tracker.add("ttfa_s", tts_first_byte(first_sentence) - t_req)
    wps = words_per_second(times, texts)
    if wps:
        tracker.add("wps", wps)


def tts_first_byte(text: str) -> float:
    t = time.perf_counter()
    with requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.eleven_voice_id}/stream",
        params={"output_format": "pcm_16000"},
        headers={"xi-api-key": settings.eleven_api_key},
        json={"text": text, "model_id": settings.eleven_model}, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1024):
            if chunk:
                now = time.perf_counter()
                break
    return now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    tracker = LatencyTracker()
    finals = asyncio.run(stream_stt(Path(a.audio), a.seconds, tracker))
    llm = AnthropicLLM(model=settings.realtime_llm_model)
    ctx = " ".join(finals)[-6000:]
    for _ in range(a.turns):
        llm_tts_turn(llm, ctx, tracker)

    rep = tracker.report()
    rep["config"] = {"stt": settings.deepgram_model, "endpointing_ms": ENDPOINTING_MS,
                     "llm": settings.realtime_llm_model, "tts": settings.eleven_model,
                     "note": "TTFA here excludes endpointing; perceived TTFA ~= endpointing + ttfa_s"}
    out = settings.output_dir / "bench" / f"{a.name or Path(a.audio).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
