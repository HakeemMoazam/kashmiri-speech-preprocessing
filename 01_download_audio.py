import os, io
import soundfile as sf
from datasets import load_dataset, Audio

ds = load_dataset("GAASH-Lab/Bolbosh", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))   # don't decode, give me raw bytes

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/text", exist_ok=True)             # NEW: folder for transcripts

for i, ex in enumerate(ds):
    if i >= 50:
        break
    a = ex["audio"]
    src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
    y, sr = sf.read(src)
    sf.write(f"data/raw/{i:05d}.wav", y, sr)

    # NEW: save the transcript with the same number as the audio file
    with open(f"data/text/{i:05d}.txt", "w", encoding="utf-8") as f:
        f.write(ex["text"])

    print(i, sr, round(len(y) / sr, 2), "sec")

print("columns available:", list(ex.keys()))