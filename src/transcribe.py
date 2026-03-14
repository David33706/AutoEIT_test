"""Audio transcription using OpenAI Whisper."""

import whisper


def load_model(model_size="medium"):
    """Load Whisper model."""
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    print("Model loaded.")
    return model


def transcribe_audio(model, filepath, skip_seconds):
    """
    Transcribe an audio file, returning only segments after skip_seconds.

    Args:
        model: Loaded Whisper model
        filepath: Path to audio file
        skip_seconds: Seconds to skip (intro/instructions)

    Returns:
        List of segment dicts with start, end, text
    """
    result = model.transcribe(
        filepath,
        language="es",
        task="transcribe",
        no_speech_threshold=0.6,
        logprob_threshold=-0.5,
        compression_ratio_threshold=2.0,
        condition_on_previous_text=False,
    )

    segments = [
        {
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "text": s["text"].strip(),
            "no_speech_prob": round(s["no_speech_prob"], 3),
            "avg_logprob": round(s["avg_logprob"], 3),
        }
        for s in result["segments"]
        if s["start"] >= skip_seconds
    ]

    return segments