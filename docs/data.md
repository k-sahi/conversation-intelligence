# Data: the AMI Meeting Corpus

~100 hours of meetings recorded at Edinburgh, IDIAP and TNO, mostly 4 participants. About two thirds are
*scenario* meetings (ES, IS, TS series): a team designing a new TV remote control over four meetings
(a = kick-off, b = functional design, c = conceptual design, d = detailed design). The rest are
non-scenario (EN, IB, IN series). Licence CC BY 4.0.

## What we use

| Annotation | File | Use |
|---|---|---|
| Word-level transcripts with times, per speaker letter A–D | `words/<M>.<S>.words.xml` | WER/cpWER reference; reference speaker turns for DER |
| Abstractive summaries: abstract, decisions, actions, problems | `abstractive/<M>.abssumm.xml` | summary references for the matcher and judge |
| Audio: Mix-Headset (IHM mix), Array1-01 (SDM), Headset-n | `amicorpus/<M>/audio/` | system input |

Parsing notes (`data/ami.py`): words with `punc="true"` and non-word elements (`<vocalsound>`, `<disfmarker>`,
`<gap>`) are excluded; reference turns merge a speaker's words separated by ≤0.3 s. For publication-comparable
DER, the community-standard RTTMs from the BUT *AMI-diarization-setup* repository can be loaded with
`read_rttm` instead of the word-derived turns.

## Splits (full-corpus-ASR partition)

* **test**: EN2002a-d, ES2004a-d, IS1009a-d, TS3003a-d, TS3007a-d
* **dev**: ES2011a-d, IS1008a-d, TS3004a-d, TS3006a-d, IB4001-4004, IB4010, IB4011
* **test-scenario**: test minus EN2002 (all have abstractive summaries)
* **dev-mini**: ES2011a, IS1008b, TS3004c — cheap iteration set

## Audio conditions and what they test

| Stream | Difficulty | Tests |
|---|---|---|
| Mix-Headset | easier | ASR + diarization of close-talk audio summed to one channel (overlap still present) |
| Array1-01 | hardest | far-field reverberation + overlap — closest to a laptop in a meeting room |
| Headset-0..3 | multichannel | channel = speaker; cross-talk filtering |

## Caveats

Scenario meetings are role-played, single-domain, and English (many non-native speakers). AMI's references
are one annotator's view; decisions/actions lists are sparse, which is why matcher precision is a lower bound
and faithfulness is judged against the transcript, not the reference.
