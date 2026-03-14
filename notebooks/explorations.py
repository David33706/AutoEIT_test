"""
Audio exploration and analysis.
Run this to inspect audio files before transcription.
Not part of the main pipeline — used for development and debugging.
"""

import librosa
import numpy as np
import os


def analyze_audio(filepath):
    """Print basic audio properties."""
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    print(f"\n{os.path.basename(filepath)}")
    print(f"  Duration:    {minutes}:{seconds:02d}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Channels:    {'mono' if y.ndim == 1 else 'stereo'}")
    print(f"  Peak volume: {abs(y).max():.3f}")
    print(f"  Mean volume: {abs(y).mean():.4f}")
    return y, sr


def energy_timeline(y, sr, start_sec, end_sec):
    """Print second-by-second energy for a time range."""
    print(f"\nENERGY TIMELINE ({start_sec//60}:{start_sec%60:02d} - {end_sec//60}:{end_sec%60:02d})")
    print("=" * 60)
    for t in range(start_sec, min(end_sec, int(len(y) / sr))):
        chunk = y[t * sr:(t + 1) * sr]
        rms = np.sqrt(np.mean(chunk ** 2))
        bar = "█" * int(rms * 500)
        print(f"  {t//60}:{t%60:02d} | {rms:.4f} | {bar}")


def detect_speech_bursts(y, sr, start_sec, threshold=0.003, min_duration=0.3):
    """Detect speech bursts using energy threshold."""
    frame_length = int(0.5 * sr)
    hop_length = int(0.25 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    is_speech = rms > threshold
    bursts = []
    in_burst = False
    start = 0

    for i, speaking in enumerate(is_speech):
        if speaking and not in_burst:
            start = times[i]
            in_burst = True
        elif not speaking and in_burst:
            end = times[i]
            if end - start > min_duration:
                bursts.append({"start": round(start, 2), "end": round(end, 2)})
            in_burst = False

    return [b for b in bursts if b["start"] >= start_sec]


def group_into_items(bursts, gap_threshold=4.0):
    """Group speech bursts into EIT items based on silence gaps."""
    items = []
    current_item = []
    prev_end = bursts[0]["start"] if bursts else 0

    for b in bursts:
        gap = b["start"] - prev_end
        if gap > gap_threshold and current_item:
            items.append(current_item)
            current_item = [b]
        else:
            current_item.append(b)
        prev_end = b["end"]

    if current_item:
        items.append(current_item)

    return items


# ============================================================
# Run exploration on all audio files
# ============================================================

if __name__ == "__main__":
    audio_dir = "data/audio"
    skip_times = {
        "038010_EIT-2A.mp3": 150,
        "038011_EIT-1A.mp3": 150,
        "038012_EIT-2A.mp3": 720,
        "038015_EIT-1A.mp3": 150,
    }

    for filename in sorted(os.listdir(audio_dir)):
        if not filename.endswith(".mp3"):
            continue

        filepath = os.path.join(audio_dir, filename)
        y, sr = analyze_audio(filepath)

        skip = skip_times.get(filename, 150)
        bursts = detect_speech_bursts(y, sr, start_sec=skip)
        items = group_into_items(bursts)

        print(f"\n  Detected {len(items)} EIT items (expecting 30)")
        no_response = sum(1 for item in items if len(item) == 1)
        print(f"  Items with no detected learner response: {no_response}")