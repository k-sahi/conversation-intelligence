"""Batch pipeline: audio -> words -> speaker turns -> utterances -> MeetingRecord.

Every stage is cached to `outputs/<meeting_id>/` so re-running evals or tweaking
the summarizer never re-pays for ASR or diarization.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .align import assign_speakers, build_utterances, make_transcript, relabel_speakers, speaker_stats
from .config import settings
from .diarization.pyannote_diar import DiarizationResult, read_rttm, write_rttm
from .observability import span
from .schemas import MeetingRecord, SpeakerTurn, Transcript, Word


@dataclass
class PipelineOptions:
    multichannel: bool = False            # IHM: one channel per speaker, skip pyannote
    num_speakers: int | None = None       # oracle count (report separately from blind runs!)
    min_speakers: int | None = 2
    max_speakers: int | None = 8
    keyterms: list[str] | None = None
    use_cache: bool = True
    skip_summary: bool = False


@dataclass
class PipelineResult:
    transcript: Transcript
    record: MeetingRecord | None
    diarization: DiarizationResult | None
    timings: dict[str, float] = field(default_factory=dict)
    out_dir: Path | None = None


class BatchPipeline:
    def __init__(self, asr=None, diarizer=None, extractor=None, output_dir: Path | None = None):
        # Components are injected so tests can pass fakes and the API can share instances.
        self._asr = asr
        self._diarizer = diarizer
        self._extractor = extractor
        self.output_dir = Path(output_dir or settings.output_dir)

    # Lazy construction keeps heavy imports (torch, pyannote) out of the API process until needed.
    @property
    def asr(self):
        if self._asr is None:
            from .asr import DeepgramBatchASR
            self._asr = DeepgramBatchASR()
        return self._asr

    @property
    def diarizer(self):
        if self._diarizer is None:
            from .diarization import PyannoteDiarizer
            self._diarizer = PyannoteDiarizer()
        return self._diarizer

    @property
    def extractor(self):
        if self._extractor is None:
            from .summarize import AnthropicLLM, MeetingExtractor
            self._extractor = MeetingExtractor(AnthropicLLM())
        return self._extractor

    # ---------------------------------------------------------------- stages
    def _asr_stage(self, audio: Path, out: Path, opts: PipelineOptions) -> list[Word]:
        from .asr import parse_deepgram_response

        cache = out / "deepgram.json"
        if opts.use_cache and cache.exists():
            resp = json.loads(cache.read_text())
        else:
            resp = self.asr.transcribe_file(audio, multichannel=opts.multichannel, keyterms=opts.keyterms)
            cache.write_text(json.dumps(resp))
        return parse_deepgram_response(resp, multichannel=opts.multichannel)

    def _diar_stage(self, audio: Path, out: Path, opts: PipelineOptions, uri: str) -> DiarizationResult:
        rttm, ex_rttm = out / "diarization.rttm", out / "diarization_exclusive.rttm"
        if opts.use_cache and rttm.exists() and ex_rttm.exists():
            return DiarizationResult(read_rttm(rttm), read_rttm(ex_rttm), model="cache")
        res = self.diarizer(audio, num_speakers=opts.num_speakers,
                            min_speakers=opts.min_speakers, max_speakers=opts.max_speakers)
        write_rttm(res.turns, uri, rttm)
        write_rttm(res.exclusive_turns, uri, ex_rttm)
        return res

    # ---------------------------------------------------------------- run
    def run(self, audio_path: str | Path, meeting_id: str | None = None,
            opts: PipelineOptions | None = None) -> PipelineResult:
        opts = opts or PipelineOptions()
        audio = Path(audio_path)
        mid = meeting_id or audio.stem.split(".")[0]
        out = self.output_dir / mid
        out.mkdir(parents=True, exist_ok=True)
        timings: dict[str, float] = {}

        with span("pipeline", meeting_id=mid, multichannel=opts.multichannel) as root:
            t = time.perf_counter()
            with span("asr.deepgram", model=settings.deepgram_model) as s:
                words = self._asr_stage(audio, out, opts)
                s.set_attribute("n_words", len(words))
            timings["asr_s"] = time.perf_counter() - t

            diar: DiarizationResult | None = None
            if opts.multichannel:
                # Channel identity is the speaker. Remove headset bleed first.
                from .align import crosstalk_filter, frame_energy_db
                try:
                    words = crosstalk_filter(words, frame_energy_db(str(audio)))
                except Exception:
                    pass
                overlapping = _turns_from_words(words)
                exclusive = overlapping
            else:
                t = time.perf_counter()
                with span("diarization.pyannote") as s:
                    diar = self._diar_stage(audio, out, opts, mid)
                    s.set_attribute("n_speakers", len(diar.speakers))
                timings["diarization_s"] = time.perf_counter() - t
                overlapping, exclusive = diar.turns, diar.exclusive_turns

            with span("align"):
                words = assign_speakers(words, exclusive, overlapping)
                utts = build_utterances(words, overlapping, max_gap_s=settings.utterance_gap_s)
                utts, mapping = relabel_speakers(utts)
                transcript = make_transcript(mid, utts, asr_model=settings.deepgram_model,
                                             diar_model=diar.model if diar else "multichannel")
            (out / "transcript.json").write_text(transcript.model_dump_json(indent=1))
            (out / "speaker_map.json").write_text(json.dumps(mapping, indent=1))

            record = None
            if not opts.skip_summary:
                t = time.perf_counter()
                with span("extract", n_utts=len(utts)):
                    record = self.extractor.run(transcript)
                record.speaker_stats = speaker_stats(utts, overlapping, mapping)
                timings["extract_s"] = time.perf_counter() - t
                (out / "record.json").write_text(record.model_dump_json(indent=1))
                (out / "record.md").write_text(render_markdown(record))
            root.set_attribute("timings", json.dumps(timings))

        (out / "timings.json").write_text(json.dumps(timings, indent=1))
        return PipelineResult(transcript, record, diar, timings, out)


def _turns_from_words(words: list[Word], gap: float = 0.5) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    last: dict[str, SpeakerTurn] = {}
    for w in sorted(words, key=lambda x: x.start):
        spk = w.speaker or "UNKNOWN"
        cur = last.get(spk)
        if cur and w.start - cur.end <= gap:
            cur.end = max(cur.end, w.end)
        else:
            cur = SpeakerTurn(speaker=spk, start=w.start, end=w.end)
            turns.append(cur)
            last[spk] = cur
    return sorted(turns, key=lambda t: t.start)


def render_markdown(r: MeetingRecord) -> str:
    lines = [f"# Meeting {r.meeting_id}", "", "## Summary", r.summary, ""]
    if r.decisions:
        lines += ["## Decisions"] + [
            f"- **{d.id}** {d.statement}" + (f" — _{d.rationale}_" if d.rationale else "")
            + f"  (evidence: {', '.join(d.evidence)}; conf {d.confidence:.2f})" for d in r.decisions] + [""]
    if r.action_items:
        lines += ["## Action items", "| ID | Task | Owner | Due | Evidence |", "|---|---|---|---|---|"]
        lines += [f"| {a.id} | {a.task} | {a.owner or '—'} | {a.due or '—'} | {', '.join(a.evidence)} |"
                  for a in r.action_items] + [""]
    if r.topics:
        lines += ["## Topics"] + [f"- {t.title}: {t.summary}" for t in r.topics] + [""]
    if r.open_questions:
        lines += ["## Open questions"] + [f"- {q}" for q in r.open_questions] + [""]
    if r.problems:
        lines += ["## Problems raised"] + [f"- {p}" for p in r.problems] + [""]
    if r.speaker_stats:
        lines += ["## Speakers", "| Speaker | Talk time (s) | Turns | Words | Overlap (s) |", "|---|---|---|---|---|"]
        lines += [f"| {s.speaker} | {s.talk_time_s} | {s.turns} | {s.words} | {s.overlap_time_s} |"
                  for s in r.speaker_stats] + [""]
    if r.warnings:
        lines += ["## Pipeline warnings"] + [f"- {w}" for w in r.warnings]
    return "\n".join(lines) + "\n"
