import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convintel.evals.der import diarization_error_rate
from convintel.evals.latency import LatencyTracker, percentile, words_per_second
from convintel.evals.text_metrics import rouge_l
from convintel.evals.wer import cpwer, edit_counts, normalize, wer
from convintel.schemas import SpeakerTurn as T


class TestNormalize(unittest.TestCase):
    def test_fillers_case_numbers(self):
        self.assertEqual(normalize("Um, OK so 25 euros!"), ["okay", "so", "twenty", "five", "euros"])

    def test_hyphen_and_equivalents(self):
        self.assertEqual(normalize("We're gonna use push-button"), ["we're", "going", "to", "use", "push", "button"])


class TestWER(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(wer("the cat sat", "the cat sat").wer, 0.0)

    def test_counts(self):
        c = edit_counts("a b c d".split(), "a x c d e".split())
        self.assertEqual((c.sub, c.dele, c.ins), (1, 0, 1))
        self.assertAlmostEqual(c.wer, 0.5)

    def test_deletion(self):
        c = edit_counts("a b c".split(), "a c".split())
        self.assertEqual((c.sub, c.dele, c.ins), (0, 1, 0))

    def test_empty_ref(self):
        self.assertEqual(edit_counts([], ["a"]).wer, 1.0)


class TestCpWER(unittest.TestCase):
    def test_permutation_invariant(self):
        ref = {"A": "hello there", "B": "good morning"}
        hyp = {"spk1": "good morning", "spk0": "hello there"}
        c, mapping = cpwer(ref, hyp)
        self.assertEqual(c.wer, 0.0)
        self.assertEqual(mapping, {"A": "spk0", "B": "spk1"})

    def test_attribution_error_counted(self):
        # same words overall, but one word given to the wrong speaker -> WER 0, cpWER > 0
        ref = {"A": "yes we agree", "B": "great"}
        hyp = {"x": "yes we", "y": "agree great"}
        self.assertEqual(wer("yes we agree great", "yes we agree great").wer, 0.0)
        c, _ = cpwer(ref, hyp)
        self.assertEqual(c.errors, 2)  # one deletion in A-stream + one insertion in B-stream

    def test_missing_speaker(self):
        c, _ = cpwer({"A": "a b", "B": "c d"}, {"x": "a b"})
        self.assertEqual(c.dele, 2)


class TestDER(unittest.TestCase):
    def test_perfect_with_relabel(self):
        ref = [T(speaker="A", start=0, end=2), T(speaker="B", start=2, end=4)]
        hyp = [T(speaker="s1", start=0, end=2), T(speaker="s2", start=2, end=4)]
        r = diarization_error_rate(ref, hyp)
        self.assertAlmostEqual(r.der, 0.0, places=3)
        self.assertEqual(r.mapping, {"A": "s1", "B": "s2"})

    def test_confusion(self):
        ref = [T(speaker="A", start=0, end=2), T(speaker="B", start=2, end=4)]
        hyp = [T(speaker="s1", start=0, end=4)]
        r = diarization_error_rate(ref, hyp)
        self.assertAlmostEqual(r.der, 0.5, places=2)
        self.assertAlmostEqual(r.confusion, 2.0, places=2)

    def test_overlap_miss(self):
        # 1s of overlap; a system that outputs only one speaker there is charged a miss
        ref = [T(speaker="A", start=0, end=3), T(speaker="B", start=2, end=4)]
        hyp = [T(speaker="x", start=0, end=3), T(speaker="y", start=3, end=4)]
        r = diarization_error_rate(ref, hyp)
        self.assertAlmostEqual(r.miss, 1.0, places=2)
        self.assertAlmostEqual(r.der, 1.0 / 5.0, places=2)
        self.assertAlmostEqual(r.overlap_share, 2.0 / 5.0, places=2)
        skip = diarization_error_rate(ref, hyp, skip_overlap=True)
        self.assertAlmostEqual(skip.der, 0.0, places=2)

    def test_false_alarm_and_collar(self):
        ref = [T(speaker="A", start=1, end=3)]
        hyp = [T(speaker="x", start=0.9, end=3.1)]
        self.assertGreater(diarization_error_rate(ref, hyp).false_alarm, 0)
        self.assertAlmostEqual(diarization_error_rate(ref, hyp, collar=0.25).der, 0.0, places=3)


class TestLatency(unittest.TestCase):
    def test_percentile(self):
        xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertAlmostEqual(percentile(xs, 50), 5.5)
        self.assertAlmostEqual(percentile(xs, 90), 9.1)

    def test_tracker(self):
        t = LatencyTracker()
        t.mark("turn", 10.0)
        self.assertAlmostEqual(t.since("turn", "ttfa_s", 10.8), 0.8)
        self.assertIsNone(t.since("turn", "ttfa_s", 11.0))  # consumed
        self.assertEqual(t.report()["ttfa_s"]["n"], 1)

    def test_wps(self):
        self.assertAlmostEqual(words_per_second([0.0, 1.0, 2.0], ["one two ", "three four ", "five six"]), 3.0)


class TestRouge(unittest.TestCase):
    def test_rouge(self):
        self.assertEqual(rouge_l("the case is rubber", "the case is rubber")["f"], 1.0)
        self.assertLess(rouge_l("the case is rubber", "plastic was chosen")["f"], 0.3)


if __name__ == "__main__":
    unittest.main()
