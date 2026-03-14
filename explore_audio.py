import librosa
import os

audio_dir = "audio"

for filename in sorted(os.listdir(audio_dir)):
    if not filename.endswith(".mp3"):
        continue

    filepath = os.path.join(audio_dir, filename)
    y, sr = librosa.load(filepath, sr=None)  # Load at native sample rate
    duration = librosa.get_duration(y=y, sr=sr)

    minutes = int(duration // 60)
    seconds = int(duration % 60)

    print(f"\n{filename}")
    print(f"  Duration:    {minutes}:{seconds:02d}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Channels:    {'mono' if y.ndim == 1 else 'stereo'}")
    print(f"  Samples:     {len(y):,}")
    print(f"  Peak volume: {abs(y).max():.3f}")
    print(f"  Mean volume: {abs(y).mean():.4f}")