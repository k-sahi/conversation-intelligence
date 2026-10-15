"""Reconcile ASR words with diarization turns, and build utterances.

Word -> speaker assignment
--------------------------
1. Use the *exclusive* diarization (one speaker at a time). Assign each word to the
   turn it overlaps most in time.
2. If a word falls in a gap (ASR and diarization disagree on where speech is),
   snap it to the nearest turn within `max_snap_s`.
3. Otherwise keep the ASR's own label (Deepgram diarize) prefixed `dg_`, or UNKNOWN.

Overlap handling
----------------
The *overlapping* diarization tells us when 2+ speakers were active. Words spoken
in those regions get `overlap=True` and the utterance records its `co_speakers`.
The summarizer is told which lines are overlapped, so it can lower confidence
on attributions there instead of silently trusting them.

For multi-channel (IHM) audio, `crosstalk_filter` removes words that a headset
picked up from a neighbour, using per-channel energy.
"""
from __future__ import annotations

import bisect
from collections import defaultdict

from .schemas import SpeakerTurn, SpeakerStats, Transcript, Utterance, Word


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


class TurnIndex:
    """Sorted-interval lookup so assignment is O(W log T) instead of O(W*T)."""

    def __init__(self, turns: list[SpeakerTurn]):
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]
        self.max_len = max((t.duration for t in self.turns), default=0.0)

    def candidates(self, t0: float, t1: float) -> list[SpeakerTurn]:
        lo = bisect.bisect_left(self.starts, t0 - self.max_len)
        hi = bisect.bisect_right(self.starts, t1)
        return [t for t in self.turns[lo:hi] if t.end > t0 and t.start < t1]

    def nearest(self, t: float, max_dist: float) -> SpeakerTurn | None:
        i = bisect.bisect_left(self.starts, t)
        best, best_d = None, max_dist
        for j in (i - 2, i - 1, i, i + 1):
            if 0 <= j < len(self.turns):
                turn = self.turns[j]
                d = 0.0 if turn.start <= t <= turn.end else min(abs(t - turn.start), abs(t - turn.end))
                if d <= best_d:
                    best, best_d = turn, d
        return best


def assign_speakers(
    words: list[Word],
    exclusive_turns: list[SpeakerTurn],
    overlapping_turns: list[SpeakerTurn] | None = None,
    max_snap_s: float = 0.5,
) -> list[Word]:
    ex_idx = TurnIndex(exclusive_turns)
    ov_idx = TurnIndex(overlapping_turns) if overlapping_turns else None
    out: list[Word] = []
    for w in words:
        w = w.model_copy()
        cands = ex_idx.candidates(w.start, w.end)
        if cands:
            best = max(cands, key=lambda t: _overlap(w.start, w.end, t.start, t.end))
            w.speaker = best.speaker
        else:
            mid = (w.start + w.end) / 2
            near = ex_idx.nearest(mid, max_snap_s)
            if near is not None:
                w.speaker = near.speaker
            elif not w.speaker:
                w.speaker = "UNKNOWN"
        if ov_idx is not None:
            active = {t.speaker for t in ov_idx.candidates(w.start, w.end)
                      if _overlap(w.start, w.end, t.start, t.end) > 0.5 * max(w.end - w.start, 1e-3)}
            w.overlap = len(active) > 1
        out.append(w)
    return out


def build_utterances(
    words: list[Word],
    overlapping_turns: list[SpeakerTurn] | None = None,
    max_gap_s: float = 0.8,
    max_len_s: float = 30.0,
) -> list[Utterance]:
    """Group consecutive same-speaker words into utterances."""
    ov_idx = TurnIndex(overlapping_turns) if overlapping_turns else None
    groups: list[list[Word]] = []
    for w in sorted(words, key=lambda x: x.start):
        if groups:
            g = groups[-1]
            same = g[-1].speaker == w.speaker
            close = w.start - g[-1].end <= max_gap_s
            short = w.end - g[0].start <= max_len_s
            if same and close and short:
                g.append(w)
                continue
        groups.append([w])

    utts: list[Utterance] = []
    for i, g in enumerate(groups):
        start, end = g[0].start, g[-1].end
        co: set[str] = set()
        if ov_idx is not None:
            for w in g:
                if w.overlap:
                    co |= {t.speaker for t in ov_idx.candidates(w.start, w.end)}
            co.discard(g[0].speaker)
        utts.append(
            Utterance(
                id=f"u{i:04d}",
                speaker=g[0].speaker or "UNKNOWN",
                start=round(start, 3),
                end=round(end, 3),
                text=" ".join(w.text for w in g).strip(),
                words=g,
                overlap_ratio=round(sum(w.overlap for w in g) / len(g), 3),
                co_speakers=sorted(co),
            )
        )
    return utts


def relabel_speakers(utts: list[Utterance], prefix: str = "SPEAKER_") -> tuple[list[Utterance], dict[str, str]]:
    """Rename speakers in order of first appearance: SPEAKER_A, SPEAKER_B, ..."""
    mapping: dict[str, str] = {}
    for u in utts:
        if u.speaker not in mapping and u.speaker != "UNKNOWN":
            mapping[u.speaker] = f"{prefix}{chr(ord('A') + len(mapping))}" if len(mapping) < 26 else f"{prefix}{len(mapping)}"
    mapping["UNKNOWN"] = "UNKNOWN"
    new = []
    for u in utts:
        u = u.model_copy(update={"speaker": mapping.get(u.speaker, u.speaker),
                                 "co_speakers": [mapping.get(s, s) for s in u.co_speakers]})
        new.append(u)
    return new, mapping


def speaker_stats(utts: list[Utterance], overlapping_turns: list[SpeakerTurn] | None = None,
                  mapping: dict[str, str] | None = None) -> list[SpeakerStats]:
    talk = defaultdict(float)
    turns = defaultdict(int)
    nwords = defaultdict(int)
    for u in utts:
        talk[u.speaker] += u.end - u.start
        turns[u.speaker] += 1
        nwords[u.speaker] += len(u.words) or len(u.text.split())
    overlap = defaultdict(float)
    if overlapping_turns:
        overlap = _overlap_time_per_speaker(overlapping_turns)
        if mapping:
            overlap = defaultdict(float, {mapping.get(k, k): v for k, v in overlap.items()})
    return [
        SpeakerStats(speaker=s, talk_time_s=round(talk[s], 2), turns=turns[s], words=nwords[s],
                     overlap_time_s=round(overlap.get(s, 0.0), 2))
        for s in sorted(talk, key=lambda k: -talk[k])
    ]


def _overlap_time_per_speaker(turns: list[SpeakerTurn]) -> dict[str, float]:
    """Seconds each speaker spent talking while someone else also talked (sweep line)."""
    events = []
    for t in turns:
        events.append((t.start, 1, t.speaker))
        events.append((t.end, -1, t.speaker))
    events.sort(key=lambda e: (e[0], e[1]))
    active: dict[str, int] = defaultdict(int)
    out: dict[str, float] = defaultdict(float)
    prev = None
    for time, delta, spk in events:
        if prev is not None and time > prev:
            live = [s for s, c in active.items() if c > 0]
            if len(live) > 1:
                for s in live:
                    out[s] += time - prev
        active[spk] += delta
        prev = time
    return out


def crosstalk_filter(words: list[Word], channel_energy_db: dict[int, list[float]], frame_s: float = 0.01,
                     margin_db: float = 6.0) -> list[Word]:
    """Drop words a headset heard from a neighbour.

    For each word on channel c, compare c's mean energy over the word against the
    loudest other channel. Bleed is typically 10-20 dB quieter than the source
    headset; keep the word only if c is within `margin_db` of the loudest.
    """
    kept = []
    for w in words:
        if w.channel is None:
            kept.append(w)
            continue
        i0, i1 = int(w.start / frame_s), max(int(w.start / frame_s) + 1, int(w.end / frame_s))

        def mean_db(ch: int) -> float:
            seg = channel_energy_db.get(ch, [])[i0:i1]
            return sum(seg) / len(seg) if seg else -120.0

        own = mean_db(w.channel)
        loudest = max(mean_db(ch) for ch in channel_energy_db)
        if own >= loudest - margin_db:
            kept.append(w)
    return kept


def frame_energy_db(path: str, frame_s: float = 0.01) -> dict[int, list[float]]:
    """Per-channel frame energy (dBFS) from a PCM WAV, stdlib + numpy only."""
    import wave

    import numpy as np

    with wave.open(path, "rb") as wf:
        n_ch, sw, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    dtype = {2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).reshape(-1, n_ch).astype(np.float64) / float(np.iinfo(dtype).max)
    hop = int(sr * frame_s)
    m = x.shape[0] // hop
    frames = x[: m * hop].reshape(m, hop, n_ch)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    return {c: db[:, c].tolist() for c in range(n_ch)}


def make_transcript(meeting_id: str, utts: list[Utterance], asr_model: str = "", diar_model: str = "",
                    source: str = "batch") -> Transcript:
    return Transcript(
        meeting_id=meeting_id,
        duration=max((u.end for u in utts), default=0.0),
        utterances=utts,
        speakers=sorted({u.speaker for u in utts}),
        asr_model=asr_model,
        diarization_model=diar_model,
        source=source,  # type: ignore[arg-type]
    )
