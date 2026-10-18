import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MelDataset(Dataset):
    def __init__(self, files, text_dir="data/text"):
        self.files = files
        self.text_dir = text_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mel_path = self.files[i]
        mel = torch.tensor(np.load(mel_path), dtype=torch.float32)

        # same number as the spectrogram: data/mel/00000.npy -> data/text/00000.txt
        name = os.path.splitext(os.path.basename(mel_path))[0]
        with open(os.path.join(self.text_dir, name + ".txt"), encoding="utf-8") as f:
            text = f.read()

        return mel, text


def collate(batch):
    mels, texts = zip(*batch)                       # split the pairs into two groups
    longest = max(m.shape[1] for m in mels)
    padded = [torch.nn.functional.pad(m, (0, longest - m.shape[1])) for m in mels]
    return torch.stack(padded), list(texts)         # tensor + list of strings


files = sorted(glob.glob("data/mel/*.npy"))
loader = DataLoader(MelDataset(files), batch_size=16, shuffle=True, collate_fn=collate)

for mels, texts in loader:
    print(mels.shape, len(texts))                   # (16, 128, longest_in_this_batch) 16

print(texts[0])                                     # one Kashmiri transcript, to check