import os, glob
import numpy as np
import librosa
import matplotlib.pyplot as plt

os.makedirs("data/mel", exist_ok=True)

for path in sorted(glob.glob("data/raw/*.wav")):
    y, sr = librosa.load(path, sr=16000)           # load + resample to 16 kHz
    y, _ = librosa.effects.trim(y, top_db=20)      # cut silent padding
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=128)
    log_mel = np.log(mel + 1e-6)                   # log compression
    name = os.path.basename(path).replace(".wav", ".npy")
    np.save(os.path.join("data", "mel", name), log_mel)
    print(name, log_mel.shape)                     # (128, time_frames)

plt.imshow(log_mel, origin="lower", aspect="auto")
plt.colorbar()
plt.xlabel("time frames")
plt.ylabel("mel bins")
plt.show()