# AutoEIT: Automated Transcription of Spanish EIT Learner Audio

## Introduction

This project implements an automated transcription pipeline for Spanish Elicited Imitation Task (EIT) audio recordings. The EIT is a sentence-repetition task used to assess second language proficiency: learners listen to a target sentence in Spanish and attempt to repeat it. The challenge is that standard speech-to-text systems are optimized for native speakers and perform poorly on non-native speech, which features phonological variation, disfluencies, incomplete productions, and transfer effects from the learner's first language.

The pipeline takes raw audio recordings of EIT sessions and produces per-sentence transcriptions suitable for subsequent scoring.

## Approach

### Model Selection

I used OpenAI's Whisper, an open-source multilingual ASR model trained on 680,000 hours of audio. Whisper was chosen for several reasons:

- Strong multilingual support including Spanish
- Better handling of accented and non-native speech compared to most commercial APIs
- Free and open-source, making it scalable to large datasets
- Multiple model sizes allowing speed/accuracy tradeoffs

I tested both `medium` (769M parameters) and `large-v3` (1.5B parameters). Both produce comparable results for higher-proficiency learners, but neither handles very low-proficiency speech well.

### Pipeline Overview

The pipeline has three stages:

**1. Transcription** — Each audio file is passed through Whisper with the following configuration:
- Language forced to Spanish (`language="es"`) to prevent the model from switching to English during accented speech
- `condition_on_previous_text=False` to prevent hallucinations from propagating across segments
- `no_speech_threshold=0.6`, `logprob_threshold=-0.5`, and `compression_ratio_threshold=2.0` to aggressively filter hallucinated text during silent portions
- A per-participant skip time to bypass the English instruction period at the start of each recording

**2. Segment-to-item matching** — Whisper outputs variable-length segments that don't map 1:1 to the 30 EIT items. I use normalized Levenshtein similarity to match each segment to its closest target sentence, then apply greedy assignment so each target gets exactly one transcription.

**3. Excel output** — Matched transcriptions are written into the provided Excel template in column C.

### Key Design Decisions

**Forcing Spanish language:** Auto-detection would risk classifying heavily accented learner speech as English or another language. Forcing Spanish ensures consistent transcription even when the learner's production is heavily influenced by L1 transfer.

**Disabling context conditioning:** Whisper's default behavior feeds previous transcription into the next segment as context. For EIT data, this causes hallucinations to snowball — a single misheard word during silence produces cascading errors. Disabling this trades some contextual coherence for much better robustness.

**Greedy matching over sequential assignment:** I initially tried sequential approaches (assigning segments in time order), but Whisper sometimes merges two items into one segment or splits one item across multiple segments. Similarity-based matching handles these edge cases more robustly.

## Challenges

### 1. Hallucination During Silence

EIT recordings contain long silent periods while the learner listens to the next stimulus. Whisper is known to hallucinate text during silence — in my initial baseline run, the model produced repeated "Gracias" tokens during the instruction period, and generated Korean, Russian, and nonsense text during inter-item silences. The aggressive filtering parameters largely solved this, but some hallucinated content still appears for the most difficult recordings.

### 2. Stimulus vs. Learner Separation

Each EIT item contains two utterances: the pre-recorded stimulus and the learner's repetition attempt. The pipeline needs to transcribe only the learner. I explored energy-based segmentation (using RMS volume differences) to separate them, but the approach was unreliable — learner volume varied widely across participants and items, with significant overlap between stimulus and learner energy levels.

In practice, Whisper often transcribes only one of the two utterances per item. For higher-proficiency learners, it tends to capture the learner's version (identifiable by systematic errors). For ambiguous cases where the transcription matches the target perfectly (similarity = 1.0), it's unclear whether Whisper captured the stimulus or whether the learner repeated perfectly.

### 3. Very Low Proficiency Speech (Participant 038012)

Participant 038012 represents the hardest case: very low proficiency with minimal intelligible production. Audio analysis revealed that this participant produced no response at all for approximately 8 of the 30 items. For items where they did respond, their production was often so fragmentary that Whisper could not resolve it into coherent Spanish, producing hallucinated output in other languages or nonsense tokens. This is a fundamental limitation of current ASR technology when applied to speech that falls below a minimum intelligibility threshold.

### 4. Segment Alignment

Whisper's segmentation doesn't always align with item boundaries. Sometimes two items are merged into a single segment (e.g., participant 038015, item 19 contains both item 18 and 19 text). Other times a single item is split across multiple segments. The matching algorithm handles most of these cases, but imperfect alignment occasionally leads to wrong assignments.

## Evaluation

### Methodology

I evaluate transcription quality by comparing ASR output against the target sentences using normalized Levenshtein similarity. While the ideal evaluation would compare against human reference transcriptions (not available for this test), similarity to the target still provides useful signal:

- **High similarity (>0.90):** The learner repeated accurately, or Whisper captured the stimulus
- **Medium similarity (0.50–0.90):** Clear learner errors captured — the most informative range
- **Low similarity (<0.50):** Significant learner difficulty, possible ASR errors, or misalignment

### Results by Participant

| Participant | Segments | Matched | Avg Similarity | Assessment |
|---|---|---|---|---|
| 038015 | 28 | 28/30 | ~0.82 | Best result. Clear learner errors captured. |
| 038011 | 38 | 30/30 | ~0.78 | Good. Some wrong matches for hardest items. |
| 038010 | 42 | 30/30 | ~0.85 | Mixed. Many perfect matches may be stimuli. |
| 038012 | 42 | 29/30 | ~0.45 | Poor. ASR breaks down at very low proficiency. |

### Error Categories

Errors fall into several categories:

- **Correct learner transcription:** ASR accurately captures what the learner said, including their errors (e.g., "El carro no tiene pelo" for target "El carro lo tiene Pedro"). These are successes.
- **Stimulus capture:** ASR transcribes the pre-recorded stimulus instead of the learner. Indistinguishable from a perfect learner repetition without additional analysis.
- **ASR errors on learner speech:** ASR mishears the learner's production (e.g., "publicidad" for "policía"). These reduce transcription accuracy.
- **Hallucination:** ASR generates text not present in the audio. Most common during silence or with very low-proficiency speakers.
- **Misalignment:** A correct transcription is assigned to the wrong item number.

## Future Improvements

### Short-term (within project scope)

- **Segment-level audio analysis:** Use spectral features (not just RMS energy) to classify each segment as stimulus or learner. The pre-recorded stimulus likely has different frequency characteristics from live ambient speech.
- **WhisperX for word-level timestamps:** Would enable more precise segment boundaries and better stimulus/learner separation.
- **Confidence-based flagging:** Use Whisper's `no_speech_prob` and `avg_logprob` scores to automatically flag low-confidence transcriptions for human review.

### Medium-term

- **Fine-tuning on learner speech:** Collect a small dataset of human-transcribed learner recordings and fine-tune Whisper (or wav2vec2) to better handle non-native phonological patterns.
- **Ensemble methods:** Run multiple ASR models and use majority voting or confidence weighting to improve accuracy.
- **Prompt conditioning experiments:** Test whether providing the target sentence as Whisper's initial prompt improves or degrades transcription of learner errors.

### Long-term

- **Active learning pipeline:** Automatically route low-confidence items to human transcribers, then use their corrections to iteratively improve the model.
- **Proficiency-adaptive processing:** Detect learner proficiency level early in the recording and adjust ASR parameters accordingly.

## Technical Details

### Dependencies

```
openai-whisper
torch
librosa
pydub
noisereduce
soundfile
jiwer
python-Levenshtein
openpyxl
pandas
numpy
```

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py
```

Output is written to `output/AutoEIT_Transcriptions_Complete.xlsx`.

### Project Structure

```
AutoEIT_test/
├── data/
│   ├── audio/                          # MP3 files (not in repo)
│   └── AutoEIT_Sample_Audio_for_Transcribing.xlsx
├── output/
│   ├── AutoEIT_Transcriptions_Complete.xlsx
│   └── *_raw.json, *_matched.json      # Intermediate results
├── src/
│   ├── transcribe.py                   # Whisper transcription
│   ├── postprocess.py                  # Segment-to-target matching
│   └── write_output.py                 # Excel output
├── notebooks/
│   └── exploration.py                  # Audio analysis utilities
├── run_pipeline.py                     # Main entry point
├── requirements.txt
└── README.md
```
