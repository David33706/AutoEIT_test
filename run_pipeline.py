"""
AutoEIT Transcription Pipeline
Converts Spanish EIT learner audio to text transcriptions.
Usage: python run_pipeline.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.transcribe import load_model, transcribe_audio
from src.postprocess import match_segments_to_targets
from src.write_output import write_to_excel

# ============================================================
# CONFIG
# ============================================================

AUDIO_FILES = {
    "038010": "data/audio/038010_EIT-2A.mp3",
    "038011": "data/audio/038011_EIT-1A.mp3",
    "038012": "data/audio/038012_EIT-2A.mp3",
    "038015": "data/audio/038015_EIT-1A.mp3",
}

SKIP_SECONDS = {
    "038010": 150,
    "038011": 150,
    "038012": 720,
    "038015": 150,
}

SHEET_NAMES = {
    "038010": "38010-2A",
    "038011": "38011-1A",
    "038012": "38012-2A",
    "038015": "38015-1A",
}

TEMPLATE_PATH = "data/AutoEIT_Sample_Audio_for_Transcribing.xlsx"
OUTPUT_DIR = "output"
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "AutoEIT_Transcriptions_Complete.xlsx")
MODEL_SIZE = "large-v3"


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(MODEL_SIZE)
    all_results = {}

    for pid, filepath in AUDIO_FILES.items():
        print(f"\n{'='*60}")
        print(f"Processing {pid}...")

        # Step 1: Transcribe
        segments = transcribe_audio(model, filepath, SKIP_SECONDS[pid])
        print(f"  Transcribed: {len(segments)} segments")

        # Save raw transcription
        with open(os.path.join(OUTPUT_DIR, f"{pid}_raw.json"), "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        # Step 2: Match to targets
        results = match_segments_to_targets(segments)
        matched = sum(1 for r in results if r["transcription"] != "[no response detected]")
        print(f"  Matched: {matched}/30")

        # Save matched results
        with open(os.path.join(OUTPUT_DIR, f"{pid}_matched.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        all_results[pid] = results

        for r in results:
            flag = " ← MISSING" if r["transcription"] == "[no response detected]" else ""
            print(f"    {r['item']:>2}. {r['transcription']}{flag}")

    # Step 3: Write Excel
    write_to_excel(all_results, TEMPLATE_PATH, OUTPUT_XLSX, SHEET_NAMES)
    print("\nDone!")


if __name__ == "__main__":
    main()