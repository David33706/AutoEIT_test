# AutoEIT: Automated Transcription and Scoring of Spanish EIT Learner Audio

## Overview

This project implements two automated pipelines for the Spanish Elicited Imitation Task (EIT):

1. **Test I — Audio-to-text transcription:** Converts raw EIT audio recordings into per-sentence transcriptions
2. **Test II — Automated scoring:** Applies the Ortega (2000) meaning-based rubric (0–4 scale) to score learner transcriptions

Both pipelines are designed to handle the unique challenges of non-native learner speech: phonological variation, disfluencies, incomplete productions, and transfer effects from the learner's first language.

---

## Test I: Audio-to-Text Transcription

### Approach

**Model:** OpenAI Whisper (`large-v3`), chosen for its strong multilingual support and better handling of accented/non-native speech compared to commercial APIs.

**Pipeline stages:**
1. **Transcription** — Whisper ASR with parameters optimized for learner speech
2. **Post-processing** — Levenshtein similarity-based matching of segments to target sentences
3. **Output** — Results written to the provided Excel template

**Key configuration choices:**
- `language="es"` — Forces Spanish to prevent language-switching on accented speech
- `condition_on_previous_text=False` — Prevents hallucinations from cascading across segments
- `no_speech_threshold=0.6`, `logprob_threshold=-0.5`, `compression_ratio_threshold=2.0` — Aggressively filters hallucinated text during silence
- Per-participant skip times to bypass the English instruction period

### Challenges

**Hallucination during silence:** Without filtering, Whisper generated repeated tokens ("Gracias"), Korean/Russian characters, and nonsense text during inter-item silence. The filtering parameters largely solved this.

**Stimulus vs. learner separation:** Each item contains both the pre-recorded stimulus and the learner's response. Energy-based separation (RMS volume) was unreliable due to overlapping volume ranges. Perfect-match transcriptions (similarity ≥ 0.95) remain ambiguous.

**Very low proficiency (038012):** Audio analysis revealed no response for ~8 of 30 items. Where the learner did respond, speech was too fragmentary for Whisper, producing hallucinated output. This is a fundamental ASR limitation.

**Segment alignment:** Whisper occasionally merges two items into one segment or splits one across multiple segments. The matching algorithm handles most cases but occasionally misassigns.

### Results

| Participant | Segments | Matched | Avg Similarity | Assessment |
|---|---|---|---|---|
| 038015 | 28 | 28/30 | ~0.82 | Best. Clear learner errors captured. |
| 038011 | 38 | 30/30 | ~0.78 | Good. Some wrong matches for hardest items. |
| 038010 | 42 | 30/30 | ~0.85 | Mixed. Many perfect matches may be stimuli. |
| 038012 | 42 | 29/30 | ~0.45 | Poor. ASR breaks down at very low proficiency. |

### Usage

```bash
python run_pipeline.py
```

Output: `output/AutoEIT_Transcriptions_Complete.xlsx`

---

## Test II: Automated Scoring

### Approach

I implemented two complementary scoring approaches and compared them:

#### 1. Rule-Based Scoring (`src/score_rules.py`)

A custom algorithm that scores responses using:
- **Content word overlap** — fraction of target content words found in the response (with fuzzy matching via Levenshtein distance to handle pronunciation variants like "manajar" → "manejar")
- **Levenshtein similarity** — overall string-level similarity after normalization
- **Response length** — to distinguish minimal responses (score 0) from partial ones (score 1)

Key implementation details:
- **Accent stripping** via Unicode normalization ("películas" == "peliculas")
- **False-friend detection** to prevent incorrect fuzzy matches (e.g., "ducha" ≠ "lucha" despite high character similarity)
- **Annotation cleaning** — removes [pause], [gibberish], xxx, false starts, and other transcription markers before scoring
- Threshold-based assignment to rubric levels (0–4)

#### 2. LLM-Based Scoring (`src/score_llm.py`)

Uses GPT-4o-mini with the full Ortega (2000) rubric as a system prompt. Each stimulus-response pair is sent individually with instructions to return a score and brief reasoning. Temperature is set to 0.0 for deterministic output.

### Comparison Results

| Participant | Rule Avg | LLM Avg | Exact Match | Within 1 Point |
|---|---|---|---|---|
| 38001-1A | 3.07 | 2.90 | 63% | 87% |
| 38004-2A | 2.63 | 2.00 | 47% | 90% |
| 38002-2A | 1.90 | 1.17 | 33% | 87% |
| 38006-2A | 1.73 | 1.10 | 57% | 80% |

**Key findings:**
- Both approaches correctly rank participants by proficiency level (38001 > 38004 > 38002 > 38006)
- The LLM scores consistently lower (stricter meaning interpretation)
- Within-1-point agreement is 80–90%, indicating both methods track the same patterns
- The biggest disagreements occur on borderline 2/3 cases — the same boundary where human raters also disagree
- The LLM provides reasoning for each score, which is valuable for transparency and debugging

**Tradeoffs:**

| | Rule-Based | LLM-Based |
|---|---|---|
| Speed | Fast (milliseconds per item) | Slow (1-2s per item, API calls) |
| Cost | Free | ~$0.50 per 120 items |
| Reproducibility | Fully deterministic | Near-deterministic (temp=0) |
| Nuance | Limited to word overlap heuristics | Understands semantic meaning |
| Transparency | Explainable via metrics | Provides natural language reasoning |

### Challenges

**Borderline cases (2 vs. 3):** The rubric itself acknowledges ambiguity: "as a general principle in case of doubt about whether meaning has changed or not, score 2." Both automated approaches struggle here, as do human raters.

**Transcription annotations:** Human transcriptions use varied notation ([gibberish], xxx, .., false starts with -). Robust cleaning is essential — missed annotations inflate content overlap scores.

**Semantic understanding:** The rule-based approach cannot determine whether a grammatical change alters meaning (e.g., "para caras" vs. "pero caras" — substituting "for" for "but" changes meaning, but both are function words). The LLM handles this better.

### Usage

```bash
# Rule-based scoring
python run_scoring.py

# LLM-based scoring (requires OpenAI API key in .env)
python run_scoring_llm.py
```

Output: `output/AutoEIT_Scores_Complete.xlsx` and `output/scores_comparison.json`

---

## Future Improvements

### Transcription (Test I)
- **WhisperX** for word-level timestamps and better segment boundaries
- **Spectral features** (not just RMS energy) to classify stimulus vs. learner
- **Fine-tuning** Whisper on human-transcribed learner speech data
- **Confidence-based flagging** to route low-confidence items to human review

### Scoring (Test II)
- **Calibration** against human rater scores to tune rule-based thresholds
- **Ensemble approach** — combine rule-based and LLM scores (e.g., average, or use LLM only for borderline cases)
- **Fine-tuning** a smaller LLM on scored EIT data to reduce API costs while maintaining quality
- **Inter-rater reliability metrics** (Cohen's kappa) once human reference scores are available

---

## Project Structure

```
AutoEIT_test/
├── src/
│   ├── __init__.py
│   ├── transcribe.py          # Whisper transcription (Test I)
│   ├── postprocess.py         # Segment-to-target matching (Test I)
│   ├── write_output.py        # Excel output (Test I)
│   ├── score_rules.py         # Rule-based scoring (Test II)
│   └── score_llm.py           # LLM-based scoring (Test II)
├── data/
│   ├── audio/                                          # MP3 files (not in repo)
│   ├── AutoEIT_Sample_Audio_for_Transcribing.xlsx      # Test I template
│   ├── AutoEIT_Sample_Transcriptions_for_Scoring.xlsx  # Test II template
│   └── Spanish_EIT_Scoring_Rubric.docx                 # Scoring rubric
├── output/
│   ├── AutoEIT_Transcriptions_Complete.xlsx             # Test I results
│   ├── AutoEIT_Scores_Complete.xlsx                     # Test II results
│   └── scores_comparison.json                           # Rule vs. LLM comparison
├── notebooks/
│   └── explorations.py
├── run_pipeline.py            # Test I entry point
├── run_scoring.py             # Test II entry point (rule-based)
├── run_scoring_llm.py         # Test II entry point (LLM)
├── AutoEIT_Pipeline.ipynb     # Jupyter notebook (Test I)
├── AutoEIT_Pipeline.pdf       # PDF of notebook with output (Test I)
├── AutoEIT_Scoring.ipynb      # Jupyter notebook (Test II)
├── AutoEIT_Scoring.pdf        # PDF of notebook with output (Test II)
├── requirements.txt
└── README.md
```

## Dependencies

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
openai
python-dotenv
```

```bash
pip install -r requirements.txt
```
