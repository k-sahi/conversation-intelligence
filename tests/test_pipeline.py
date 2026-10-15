import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from convintel.align import (assign_speakers, build_utterances, crosstalk_filter, relabel_speakers,
                             speaker_stats)
from convintel.asr.deepgram_batch import parse_deepgram_response
from convintel.data.ami import load_reference, meetings_in, parse_abssumm
from convintel.diarization.pyannote_diar import make_exclusive, read_rttm, write_rttm
from convintel.evals.calibration import agreement_report, calibrate, quadratic_weighted_kappa, spearman
from convintel.evals.judge import ItemMatcher, PairwiseJudge, RubricJudge
from convintel.realtime.live_notes import LiveNotes, similar
from convintel.schemas import ActionItem, Decision, MeetingRecord, SpeakerTurn as T, Transcript, Utterance, Word
from convintel.summarize import FakeLLM, MeetingExtractor, chunk_utterances, support_score
from convintel.summarize.extractor import ReduceOutput
from convintel.summarize.heuristic import HeuristicLLM

FX = ROOT / "tests" / "fixtures"


def W(text, s, e, spk=None, ch=None):
    return Word(text=text, start=s, end=e, speaker=spk, channel=ch)


class TestAlign(unittest.TestCase):
    def test_assign_and_overlap_flag(self):
        words = [W("hi", 0.1, 0.4), W("there", 0.5, 0.9), W("yes", 1.1, 1.4), W("ok", 2.3, 2.5)]
        overlapping = [T(speaker="s0", start=0.0, end=1.5), T(speaker="s1", start=1.0, end=2.0)]
        exclusive = [T(speaker="s0", start=0.0, end=1.0), T(speaker="s1", start=1.0, end=2.0)]
        out = assign_speakers(words, exclusive, overlapping, max_snap_s=0.6)
        self.assertEqual([w.speaker for w in out], ["s0", "s0", "s1", "s1"])  # last word snapped (midpoint 0.4s past turn end)
        self.assertEqual([w.overlap for w in out], [False, False, True, False])

    def test_unknown_when_far(self):
        out = assign_speakers([W("x", 10, 10.2)], [T(speaker="s0", start=0, end=1)], max_snap_s=0.5)
        self.assertEqual(out[0].speaker, "UNKNOWN")

    def test_build_and_relabel(self):
        words = [W("a", 0, .2, "s1"), W("b", .3, .5, "s1"), W("c", .6, .8, "s0"), W("d", 3, 3.2, "s0")]
        utts = build_utterances(words, max_gap_s=0.8)
        self.assertEqual([u.text for u in utts], ["a b", "c", "d"])
        utts, mapping = relabel_speakers(utts)
        self.assertEqual(mapping["s1"], "SPEAKER_A")
        self.assertEqual(utts[1].speaker, "SPEAKER_B")

    def test_speaker_stats_overlap(self):
        utts = [Utterance(id="u0", speaker="A", start=0, end=3, text="x y"),
                Utterance(id="u1", speaker="B", start=2, end=4, text="z")]
        st = {s.speaker: s for s in speaker_stats(utts, [T(speaker="A", start=0, end=3), T(speaker="B", start=2, end=4)])}
        self.assertAlmostEqual(st["A"].overlap_time_s, 1.0)
        self.assertAlmostEqual(st["B"].talk_time_s, 2.0)

    def test_crosstalk(self):
        energy = {0: [-20.0] * 100, 1: [-38.0] * 100}   # ch1 hears ch0 at -18 dB -> bleed
        words = [W("hi", 0.1, 0.3, "ch0", 0), W("hi", 0.1, 0.3, "ch1", 1)]
        kept = crosstalk_filter(words, energy)
        self.assertEqual([w.channel for w in kept], [0])

    def test_make_exclusive(self):
        ex = make_exclusive([T(speaker="A", start=0, end=2), T(speaker="B", start=1, end=3)])
        self.assertEqual([(t.speaker, t.start, t.end) for t in ex], [("A", 0.0, 1.0), ("B", 1.0, 3.0)])


class TestIO(unittest.TestCase):
    def test_deepgram_parse(self):
        words = parse_deepgram_response(json.loads((FX / "deepgram_FX001a.json").read_text()))
        self.assertGreater(len(words), 50)
        self.assertTrue(all(w.speaker.startswith("dg_") for w in words))

    def test_deepgram_multichannel(self):
        resp = {"results": {"channels": [
            {"alternatives": [{"words": [{"word": "a", "start": 1, "end": 1.2}]}]},
            {"alternatives": [{"words": [{"word": "b", "start": 0.5, "end": 0.7}]}]}]}}
        words = parse_deepgram_response(resp, multichannel=True)
        self.assertEqual([(w.text, w.speaker, w.channel) for w in words], [("b", "ch1", 1), ("a", "ch0", 0)])

    def test_rttm_roundtrip(self):
        turns = read_rttm(FX / "diarization_FX001a.rttm")
        with tempfile.TemporaryDirectory() as d:
            write_rttm(turns, "FX001a", Path(d) / "x.rttm")
            again = read_rttm(Path(d) / "x.rttm")
        self.assertEqual(len(turns), len(again))
        self.assertAlmostEqual(turns[2].end, again[2].end, places=3)

    def test_ami_reference(self):
        ref = load_reference("FX001a", FX / "ami")
        self.assertEqual(ref.speakers, ["A", "B", "C"])
        self.assertNotIn(".", ref.text().split())         # punctuation tokens excluded
        self.assertEqual(len(ref.decisions), 2)
        self.assertTrue(ref.has_summary())
        turns = ref.turns()
        self.assertEqual(len(turns), 8)
        self.assertEqual(len(parse_abssumm(FX / "ami/abstractive/FX001a.abssumm.xml")["actions"]), 2)

    def test_splits(self):
        self.assertEqual(len(meetings_in("test")), 20)
        self.assertEqual(meetings_in("ES2002a, ES2002b"), ["ES2002a", "ES2002b"])


def _transcript(lines):
    utts = [Utterance(id=f"u{i:04d}", speaker=s, start=i * 3.0, end=i * 3.0 + 2.5, text=t)
            for i, (s, t) in enumerate(lines)]
    return Transcript(meeting_id="M", duration=len(lines) * 3.0, utterances=utts, speakers=sorted({s for s, _ in lines}))


class TestExtractor(unittest.TestCase):
    def test_chunking_overlap_and_coverage(self):
        tr = _transcript([("A", "x" * 100)] * 50)
        wins = chunk_utterances(tr.utterances, max_chars=1300, overlap=2)
        ids = {u.id for w in wins for u in w}
        self.assertEqual(len(ids), 50)                         # nothing dropped
        self.assertEqual(wins[0][-2:], wins[1][:2])             # overlap preserved

    def test_support_score(self):
        self.assertGreater(support_score("case made of rubber", ["we go with rubber for the case"]), 0.6)
        self.assertLess(support_score("budget approved for marketing", ["we go with rubber"]), 0.2)

    def test_grounding_drops_hallucinations(self):
        tr = _transcript([("A", "let's go with the rubber case"), ("B", "I will send the cost estimate friday")])

        def handler(system, user, schema):
            if schema.__name__ == "ChunkExtraction":
                return {"notes": "n"}
            return {"summary": "s", "decisions": [
                {"statement": "Case will be rubber", "evidence": ["u0000"]},
                {"statement": "Budget doubled for marketing", "evidence": ["u0000"]},   # unsupported
                {"statement": "Use titanium", "evidence": ["u9999"]}],                   # bad id
                "action_items": [{"task": "Send cost estimate", "owner": "B", "evidence": ["u0001"]}]}

        rec = MeetingExtractor(FakeLLM(handler)).run(tr)
        self.assertEqual([d.statement for d in rec.decisions], ["Case will be rubber"])
        self.assertEqual(rec.decisions[0].id, "D1")
        self.assertEqual(len(rec.action_items), 1)
        self.assertEqual(len(rec.warnings), 2)

    def test_heuristic_end_to_end(self):
        import demo_offline
        with tempfile.TemporaryDirectory() as d:
            res, row = demo_offline.main(Path(d))
        self.assertTrue(any("rubber" in x.statement for x in res.record.decisions))
        self.assertEqual(len(res.record.action_items), 2)
        self.assertLess(row["asr"]["wer"]["wer"], 0.1)
        self.assertGreaterEqual(row["asr"]["cpwer"]["wer"], row["asr"]["wer"]["wer"])
        self.assertTrue(any(u.overlap_ratio > 0 for u in res.transcript.utterances))


class TestJudges(unittest.TestCase):
    def setUp(self):
        self.tr = _transcript([("A", "we go with rubber"), ("B", "I will send costs")])
        self.rec = MeetingRecord(meeting_id="M", summary="Rubber chosen.",
                                 decisions=[Decision(id="D1", statement="Rubber case", evidence=["u0000"])],
                                 action_items=[ActionItem(id="A1", task="Send costs", owner="B", evidence=["u0001"])])

    def test_rubric_judge_median(self):
        scores = iter([5, 4, 3, 2, 4, 4, 3, 2, 3, 4, 3, 2])

        def handler(system, user, schema):
            self.assertIn("faithfulness", system)
            self.assertIn("u0000", user)       # evidence passed to judge
            return {"scores": [{"dimension": d, "analysis": "ok", "score": next(scores)}
                               for d in ["faithfulness", "decision_coverage", "action_item_quality", "summary_usefulness"]]}

        out = RubricJudge(FakeLLM(handler), n_samples=3).score(self.rec, self.tr, ["abs"], ["Rubber"], ["Costs"])
        self.assertEqual(out["scores"]["faithfulness"], 4.0)   # median of 5,4,3
        self.assertEqual(out["spread"]["faithfulness"], 2)

    def test_item_matcher(self):
        def handler(system, user, schema):
            return {"ref_matches": [{"ref_index": 0, "system_index": 0}, {"ref_index": 1, "system_index": None}],
                    "sys_support": [{"system_index": 0, "supported": True}]}
        m = ItemMatcher(FakeLLM(handler)).match(["rubber case", "kinetic battery"], ["Rubber case"])
        self.assertEqual((m["recall"], m["precision"]), (0.5, 1.0))

    def test_pairwise_position_bias(self):
        # A judge that always prefers whatever is shown first -> inconsistent -> TIE
        always_first = FakeLLM(lambda s, u, schema: {"reasoning": "", "winner": "A"})
        out = PairwiseJudge(always_first).compare(self.rec, self.rec, "ref")
        self.assertEqual(out["winner"], "TIE")
        self.assertFalse(out["order_consistent"])


class TestCalibration(unittest.TestCase):
    def test_qwk(self):
        self.assertAlmostEqual(quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0)
        self.assertLess(quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), 0)
        near = quadratic_weighted_kappa([1, 2, 3, 4, 5, 3], [1, 2, 3, 5, 5, 3])
        far = quadratic_weighted_kappa([1, 2, 3, 4, 5, 3], [1, 2, 3, 1, 5, 3])
        self.assertGreater(near, far)                          # off-by-1 penalised less than off-by-3

    def test_spearman_ties(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 1, 2], [1, 1, 2]), 1.0)

    def test_bias_and_gate(self):
        rep = agreement_report([3, 3, 4, 2], [4, 4, 5, 3])
        self.assertEqual(rep["bias"], 1.0)
        human = [{"item_id": i, "dimension": "faithfulness", "score": s, "annotator": a}
                 for i, s in enumerate([1, 2, 3, 4, 5, 3]) for a in ("h1", "h2")]
        judge = [{"item_id": i, "dimension": "faithfulness", "score": s} for i, s in enumerate([1, 2, 3, 4, 5, 3])]
        out = calibrate(human, judge)
        self.assertTrue(out["passed"])
        self.assertEqual(out["dimensions"]["faithfulness"]["inter_annotator_qwk"], 1.0)
        lenient = [dict(r, score=min(5, r["score"] + 1)) for r in judge]
        self.assertFalse(calibrate(human, lenient)["passed"])  # +1 bias fails the gate


class TestLiveNotes(unittest.TestCase):
    def test_incremental_merge_and_recap(self):
        notes = LiveNotes("room", MeetingExtractor(HeuristicLLM()), update_every_utts=2)
        notes.add_utterance("alice", "we go with rubber for the case", 0, 2)
        self.assertFalse(notes.due())
        notes.add_utterance("bob", "I will send the cost estimate by friday", 2, 4)
        self.assertTrue(notes.due())
        notes.update()
        notes.add_utterance("alice", "so we go with rubber for the case then", 4, 6)
        notes.add_utterance("carol", "what about the screen", 6, 8)
        notes.update()
        snap = notes.snapshot()
        self.assertEqual(len(snap.decisions), 1)               # restated decision merged
        self.assertEqual(snap.action_items[0].owner, "bob")
        self.assertIn("bob will", notes.recap_text())
        final = notes.finalize()
        self.assertGreaterEqual(len(final.decisions), 1)

    def test_similar(self):
        self.assertTrue(similar("we go with rubber", "so we go with rubber"))
        self.assertFalse(similar("rubber case", "titanium shell"))


class TestAPI(unittest.TestCase):
    def test_api_import(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        from fastapi.testclient import TestClient

        from convintel.api.main import app
        c = TestClient(app)
        self.assertEqual(c.get("/healthz").status_code, 200)
        self.assertEqual(c.get("/v1/meetings/doesnotexist").status_code, 404)


if __name__ == "__main__":
    unittest.main()
