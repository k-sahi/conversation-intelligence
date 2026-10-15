"""LiveKit Agents worker: a meeting note-taker that joins a room.

What it does
  * Transcribes EVERY participant with Deepgram Nova-3 streaming (one STT stream per
    participant track -> speaker attribution by identity, no diarization model needed).
  * Every ~45 s updates the live meeting record (decisions / actions) with the same
    extractor as batch, and publishes it to the room on the `meeting.notes` text topic.
  * Answers spoken questions ("what did we decide about the case?") through an
    AgentSession: Deepgram STT -> Claude -> ElevenLabs Eleven v3 Conversational TTS.
  * On "recap", speaks a short recap with ElevenLabs.
  * On disconnect, runs the full batch extraction and stores the final record.
  * Records TTFO / TTFW / WPS / TTFA into a LatencyTracker and Arize.

Run locally:   python -m convintel.realtime.agent dev
Deploy:        lk agent create   (uses Dockerfile.agent + livekit.toml)

Written against livekit-agents ~=1.8 (AgentServer API). The framework moves fast:
if an import fails, check the version pinned in requirements-realtime.txt.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from ..config import settings
from ..evals.latency import LatencyTracker
from ..summarize import AnthropicLLM, MeetingExtractor
from .live_notes import LiveNotes

logger = logging.getLogger("convintel.agent")

try:
    from livekit import agents, rtc
    from livekit.agents import Agent, AgentServer, AgentSession, JobContext, RunContext, function_tool, room_io
    from livekit.agents import stt as lk_stt
    from livekit.plugins import anthropic as lk_anthropic
    from livekit.plugins import deepgram, elevenlabs, silero
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install -r requirements-realtime.txt") from exc

NOTES_TOPIC = "meeting.notes"
RECAP_WORDS = ("recap", "summarize", "summarise", "what did we decide")


class NoteTaker(Agent):
    def __init__(self, notes: LiveNotes):
        self.notes = notes
        super().__init__(instructions=(
            "You are the meeting's note-taker. You stay silent unless someone addresses you "
            "as 'Notes' or asks for a recap. Answer from the meeting record only, using the "
            "get_meeting_record tool. Keep spoken answers under three sentences, no lists, "
            "no symbols, no markdown. If the record does not contain the answer, say so."))

    @function_tool()
    async def get_meeting_record(self, context: RunContext) -> str:
        """Return the current meeting record: decisions, action items and open questions."""
        r = self.notes.snapshot()
        return json.dumps({
            "decisions": [d.statement for d in r.decisions],
            "action_items": [{"task": a.task, "owner": a.owner, "due": a.due} for a in r.action_items],
            "open_questions": r.open_questions,
            "recent_summary": r.summary,
        })


class MultiUserTranscriber:
    """One Deepgram streaming session per remote participant audio track."""

    def __init__(self, ctx: JobContext, notes: LiveNotes, tracker: LatencyTracker, on_final=None):
        self.ctx, self.notes, self.tracker, self.on_final = ctx, notes, tracker, on_final
        self.origin = time.perf_counter()   # meeting t=0 on the perf_counter clock
        self.tasks: set[asyncio.Task] = set()
        self.stt = deepgram.STT(model=settings.deepgram_model, interim_results=True, smart_format=True,
                                punctuate=True, filler_words=True)

    def start(self) -> None:
        @self.ctx.room.on("track_subscribed")
        def _on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                task = asyncio.create_task(self._run(track, participant.identity))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

    async def _run(self, track: rtc.Track, speaker: str) -> None:
        stream = self.stt.stream()
        audio = rtc.AudioStream(track)
        stream_start = time.perf_counter()
        seen_interim_for: set[float] = set()

        async def push():
            async for ev in audio:
                stream.push_frame(ev.frame)
            stream.end_input()

        pusher = asyncio.create_task(push())
        try:
            async for ev in stream:
                if not ev.alternatives:
                    continue
                alt = ev.alternatives[0]
                now = time.perf_counter()
                seg_start = getattr(alt, "start_time", 0.0) or 0.0
                seg_end = getattr(alt, "end_time", 0.0) or 0.0
                if ev.type == lk_stt.SpeechEventType.INTERIM_TRANSCRIPT and alt.text.strip():
                    key = round(seg_start, 1)
                    if key not in seen_interim_for:
                        seen_interim_for.add(key)
                        # TTFO: audio time of segment onset (on the stream clock) -> first interim seen
                        self.tracker.add("ttfo_s", now - (stream_start + seg_start))
                elif ev.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT and alt.text.strip():
                    self.tracker.add("stt_final_lag_s", now - (stream_start + seg_end))
                    offset = stream_start - self.origin
                    u = self.notes.add_utterance(speaker, alt.text, offset + seg_start, offset + seg_end)
                    if self.on_final:
                        await self.on_final(u)
        finally:
            pusher.cancel()
            await stream.aclose()


server = AgentServer()


@server.rtc_session(agent_name="meeting-notes")
async def entrypoint(ctx: JobContext):
    meeting_id = ctx.room.name
    tracker = LatencyTracker()
    notes = LiveNotes(meeting_id, MeetingExtractor(AnthropicLLM(model=settings.extraction_model)))

    session = AgentSession(
        stt=deepgram.STT(model=settings.deepgram_model),
        llm=lk_anthropic.LLM(model=settings.realtime_llm_model),
        tts=elevenlabs.TTS(voice_id=settings.eleven_voice_id, model=settings.eleven_model),
        vad=silero.VAD.load(),
    )
    _wire_metrics(session, tracker)

    async def publish_notes():
        rec = notes.snapshot()
        try:
            await ctx.room.local_participant.send_text(rec.model_dump_json(), topic=NOTES_TOPIC)
        except Exception as exc:  # older rtc versions: fall back to data packets
            logger.debug("send_text failed (%s), using publish_data", exc)
            await ctx.room.local_participant.publish_data(rec.model_dump_json().encode(), topic=NOTES_TOPIC)

    async def on_final(u):
        if any(w in u.text.lower() for w in RECAP_WORDS):
            tracker.mark("recap_trigger")
            await session.say(notes.recap_text(), allow_interruptions=True)
        if notes.due():
            loop = asyncio.get_running_loop()
            t = time.perf_counter()
            await loop.run_in_executor(None, notes.update)
            tracker.add("notes_update_s", time.perf_counter() - t)
            await publish_notes()

    transcriber = MultiUserTranscriber(ctx, notes, tracker, on_final=on_final)
    transcriber.start()

    @session.on("agent_state_changed")
    def _on_state(ev):
        if getattr(ev, "new_state", None) == "speaking":
            tracker.since("recap_trigger", "ttfa_recap_s")

    async def shutdown():
        out = settings.output_dir / meeting_id
        out.mkdir(parents=True, exist_ok=True)
        tracker.dump(out / "realtime_latency.json")
        if notes.utterances:
            (out / "realtime_transcript.json").write_text(notes._transcript().model_dump_json(indent=1))
            loop = asyncio.get_running_loop()
            final = await loop.run_in_executor(None, notes.finalize)
            (out / "record.json").write_text(final.model_dump_json(indent=1))
        logger.info("latency %s", json.dumps(tracker.report()))

    ctx.add_shutdown_callback(shutdown)

    await session.start(room=ctx.room, agent=NoteTaker(notes),
                        room_options=room_io.RoomOptions(close_on_disconnect=False))
    await session.say("Hi, I'm taking notes. Say 'Notes, recap' any time for a summary.")


def _wire_metrics(session: AgentSession, tracker: LatencyTracker) -> None:
    """Map LiveKit's per-component metrics onto our four latency definitions."""
    per_turn: dict[str, dict] = {}

    @session.on("metrics_collected")
    def _on_metrics(ev):
        m = ev.metrics
        kind = type(m).__name__
        sid = getattr(m, "speech_id", None) or "na"
        turn = per_turn.setdefault(sid, {})
        if kind == "EOUMetrics":
            turn["eou"] = m.end_of_utterance_delay
        elif kind == "LLMMetrics":
            tracker.add("ttfw_s", m.ttft)                          # time to first word (token)
            if getattr(m, "tokens_per_second", 0):
                tracker.add("llm_tokens_per_s", m.tokens_per_second)
                tracker.add("wps_est", m.tokens_per_second * 0.75)  # ~0.75 words per token for English
            turn["ttft"] = m.ttft
        elif kind == "TTSMetrics":
            tracker.add("tts_ttfb_s", m.ttfb)
            turn["ttfb"] = m.ttfb
        if {"eou", "ttft", "ttfb"} <= turn.keys():
            tracker.add("ttfa_s", turn["eou"] + turn["ttft"] + turn["ttfb"])  # time to first audio
            per_turn.pop(sid, None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
