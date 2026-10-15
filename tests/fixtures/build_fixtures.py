"""Builds a tiny synthetic AMI-format meeting (FX001a) used by tests and the offline demo.

Writes:
  ami/words/FX001a.{A,B,C}.words.xml   reference words (NXT format, like AMI)
  ami/abstractive/FX001a.abssumm.xml   reference abstract / decisions / actions
  deepgram_FX001a.json                 a Deepgram-shaped response with realistic ASR errors
  diarization_FX001a.rttm              an imperfect diarization with one overlap
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

SCRIPT = [  # (speaker, start, text)
    ("A", 0.0, "okay so let's decide on the remote control case material"),
    ("B", 3.6, "I think we should use a rubber case it feels nice"),
    ("C", 6.2, "yeah rubber sounds good to me"),
    ("A", 9.0, "alright we go with rubber and the kinetic battery"),
    ("B", 12.4, "I will send the cost estimate by Friday"),
    ("C", 15.2, "what about the LCD screen"),
    ("A", 17.4, "we will discuss the screen at the next meeting"),
    ("C", 20.6, "fine I will prepare some screen options"),
]
WORD_DUR, GAP = 0.28, 0.05


def timed_words():
    out = []
    for spk, start, text in SCRIPT:
        t = start
        for w in text.split():
            out.append((spk, round(t, 2), round(t + WORD_DUR, 2), w))
            t += WORD_DUR + GAP
    return out


def write_ami(words):
    wd = HERE / "ami" / "words"
    wd.mkdir(parents=True, exist_ok=True)
    for spk in "ABC":
        rows = [f'  <w nite:id="FX001a.{spk}.w{i}" starttime="{s}" endtime="{e}">{w}</w>'
                for i, (sp, s, e, w) in enumerate(words) if sp == spk]
        rows.append(f'  <w nite:id="FX001a.{spk}.wp" starttime="30.0" endtime="30.0" punc="true">.</w>')
        rows.append(f'  <vocalsound nite:id="FX001a.{spk}.v0" starttime="31.0" endtime="31.5" type="laugh"/>')
        (wd / f"FX001a.{spk}.words.xml").write_text(
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
            '<nite:root nite:id="FX001a.words" xmlns:nite="http://nite.sourceforge.net/">\n'
            + "\n".join(rows) + "\n</nite:root>\n")
    ad = HERE / "ami" / "abstractive"
    ad.mkdir(parents=True, exist_ok=True)
    (ad / "FX001a.abssumm.xml").write_text(
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<nite:root nite:id="FX001a.abssumm" xmlns:nite="http://nite.sourceforge.net/">\n'
        ' <abstract nite:id="a"><sentence nite:id="s1">The team chose the case material for the remote control.'
        '</sentence><sentence nite:id="s2">The LCD screen was deferred to the next meeting.</sentence></abstract>\n'
        ' <decisions nite:id="d"><sentence nite:id="s3">The case will be made of rubber.</sentence>'
        '<sentence nite:id="s4">The remote will use a kinetic battery.</sentence></decisions>\n'
        ' <actions nite:id="ac"><sentence nite:id="s5">The industrial designer will send a cost estimate by Friday.'
        '</sentence><sentence nite:id="s6">The user interface designer will prepare screen options.</sentence></actions>\n'
        ' <problems nite:id="p"><sentence nite:id="s7">Whether to include an LCD screen is unresolved.</sentence></problems>\n'
        '</nite:root>\n')


def write_deepgram(words):
    # Inject typical ASR errors: a substitution, a deletion, an inserted filler, numerals.
    subs = {"kinetic": "kinetics", "Friday": "Friday."}
    dg_words = []
    for i, (spk, s, e, w) in enumerate(words):
        if w == "nice":          # deletion
            continue
        if w == "the" and i == 37:
            continue
        dg_words.append({"word": w.lower(), "punctuated_word": subs.get(w, w), "start": s, "end": e,
                         "confidence": 0.93, "speaker": {"A": 0, "B": 1, "C": 2}[spk]})
        if w == "rubber" and spk == "B":  # insertion of a filler
            dg_words.append({"word": "um", "punctuated_word": "um", "start": e + 0.01, "end": e + 0.02,
                             "confidence": 0.5, "speaker": 1})
    resp = {"metadata": {"model_info": {"name": "nova-3"}},
            "results": {"channels": [{"alternatives": [{"transcript": " ".join(w["punctuated_word"] for w in dg_words),
                                                        "words": dg_words}]}]}}
    (HERE / "deepgram_FX001a.json").write_text(json.dumps(resp, indent=1))


def write_rttm():
    # pyannote-style labels; C's first turn starts early (overlaps B) and B's 2nd turn is cut short.
    turns = [("SPEAKER_00", 0.0, 3.5), ("SPEAKER_01", 3.6, 6.6), ("SPEAKER_02", 6.0, 8.3),
             ("SPEAKER_00", 9.0, 11.6), ("SPEAKER_01", 12.4, 14.7), ("SPEAKER_02", 15.2, 17.0),
             ("SPEAKER_00", 17.4, 20.5), ("SPEAKER_02", 20.6, 23.0)]
    lines = [f"SPEAKER FX001a 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>" for spk, s, e in turns]
    (HERE / "diarization_FX001a.rttm").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    w = timed_words()
    write_ami(w)
    write_deepgram(w)
    write_rttm()
    print("fixtures written to", HERE)
