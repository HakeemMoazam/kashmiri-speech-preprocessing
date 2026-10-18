import csv
import glob
import os

import librosa
import numpy as np
import soundfile as sf

# ---- Thresholds: change these after looking at your own clips ----
MIN_DURATION_SEC = 1.0      # shorter than this = "too short"
MIN_LOUDNESS_DB = -45.0     # average loudness below this = "near silent"
MIN_SPEECH_RATIO = 0.30     # less than 30% of the clip is speech = "mostly silence"
CLIP_LEVEL = 0.99           # a sample this close to the maximum counts as "at the limit"
MAX_CLIPPED_FRACTION = 0.001  # more than 0.1% of samples at the limit = "clipped"
TRIM_TOP_DB = 20            # same silence rule as the pipeline

rows = []
for wav_path in sorted(glob.glob("data/raw/*.wav")):
    name = os.path.splitext(os.path.basename(wav_path))[0]
    y, sr = sf.read(wav_path)                     # samples are floats between -1 and 1
    if y.ndim > 1:                                # stereo -> keep the loudest channel for clipping
        y_clip = np.abs(y).max(axis=1)
        y = y.mean(axis=1)
    else:
        y_clip = np.abs(y)

    duration = len(y) / sr
    rms = np.sqrt(np.mean(y ** 2))
    loudness_db = 20 * np.log10(rms + 1e-9)       # average loudness in dB (0 = max, more negative = quieter)
    clipped_fraction = float(np.mean(y_clip >= CLIP_LEVEL))
    trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    speech_ratio = len(trimmed) / max(len(y), 1)

    txt_path = os.path.join("data/text", name + ".txt")
    text = open(txt_path, encoding="utf-8").read().strip() if os.path.exists(txt_path) else ""
    chars_per_sec = len(text) / duration if duration > 0 else 0

    flags = []
    if duration < MIN_DURATION_SEC:
        flags.append("too_short")
    if loudness_db < MIN_LOUDNESS_DB:
        flags.append("near_silent")
    if speech_ratio < MIN_SPEECH_RATIO:
        flags.append("mostly_silence")
    if clipped_fraction > MAX_CLIPPED_FRACTION:
        flags.append("clipped")
    if not text:
        flags.append("no_text")

    rows.append({
        "clip": name,
        "duration_s": round(duration, 2),
        "loudness_db": round(loudness_db, 1),
        "speech_ratio": round(speech_ratio, 2),
        "clipped_pct": round(clipped_fraction * 100, 3),
        "chars_per_sec": round(chars_per_sec, 1),
        "flags": ";".join(flags),
    })

# ---- Report ----
print(f"{'clip':<8}{'dur(s)':>8}{'dB':>8}{'speech':>8}{'clip%':>8}{'ch/s':>7}  flags")
for r in rows:
    print(f"{r['clip']:<8}{r['duration_s']:>8}{r['loudness_db']:>8}{r['speech_ratio']:>8}"
          f"{r['clipped_pct']:>8}{r['chars_per_sec']:>7}  {r['flags']}")

flagged = [r for r in rows if r["flags"]]
print(f"\nChecked {len(rows)} clips, flagged {len(flagged)}.")
for r in flagged:
    print(f"  {r['clip']}: {r['flags']}")

os.makedirs("data", exist_ok=True)
with open("data/quality_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("Saved data/quality_report.csv")