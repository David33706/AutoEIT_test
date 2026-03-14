import whisper
import json

model = whisper.load_model("medium")
print("Model loaded.")

# Start with the shortest file
result = model.transcribe(
    "data/audio/038015_EIT-1A.mp3",
    language="es",
    task="transcribe",
    verbose=True  # Prints each segment as it's transcribed
)

# Save full output for inspection
with open("data/baseline_038015.json", "w", encoding="utf-8") as f:
    json.dump({
        "text": result["text"],
        "segments": [
            {
                "id": s["id"],
                "start": s["start"],
                "end": s["end"],
                "text": s["text"]
            }
            for s in result["segments"]
        ]
    }, f, ensure_ascii=False, indent=2)

print("\n\nFull transcription saved to data/baseline_038015.json")
print(f"\nTotal segments: {len(result['segments'])}")
print(f"\nFull text:\n{result['text']}")