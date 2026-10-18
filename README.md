# Kashmiri Speech Preprocessing Pipeline

Audio preprocessing pipeline for Kashmiri text-to-speech: resampling, silence trimming, log-Mel feature extraction, and a PyTorch `DataLoader` that pairs each spectrogram with its transcript. Demonstrated on a 50-clip sample of the [GAASH-Lab/Bolbosh](https://huggingface.co/datasets/GAASH-Lab/Bolbosh) dataset.

## Pipeline

| Script | Step | Output |
|---|---|---|
| `01_download_audio.py` | Stream clips from Hugging Face; save audio as WAV and transcripts as UTF-8 text | `data/raw/*.wav`, `data/text/*.txt` |
| `02_extract_log_mel.py` | Resample to 16 kHz, trim silence, compute log-Mel spectrograms | `data/mel/*.npy` |
| `03_dataset_dataloader.py` | `Dataset` returning (spectrogram, transcript), zero-padding collate function, `DataLoader` | Padded batches with transcripts |
| `04_quality_check.py` | Flag clips that are too short, near silent, mostly silence, clipped, or missing text | `data/quality_report.csv` |

## Configuration

| Parameter | Value |
|---|---|
| Sampling rate | 16 kHz |
| Silence trim | `top_db=20` |
| Mel bins | 128 |
| `n_fft` | 1024 (64 ms) |
| `hop_length` | 256 (16 ms) |
| Log offset | `1e-6` |
| Batch size | 16 (shuffled) |

Example batch: spectrograms of shape `[16, 128, 711]` (clips, mel bins, time frames) and a list of 16 transcripts.

## Quality check

Default thresholds, set at the top of `04_quality_check.py`: minimum duration 1.0 s, minimum loudness -45 dB, minimum speech ratio 30%, and at most 0.1% of samples at the maximum level.

On the 50-clip sample, 6 clips were flagged as too short; none were flagged as silent, clipped, or missing text.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

python 01_download_audio.py
python 02_extract_log_mel.py
python 03_dataset_dataloader.py
python 04_quality_check.py
```

## Data

[GAASH-Lab/Bolbosh](https://huggingface.co/datasets/GAASH-Lab/Bolbosh), licensed CC BY 4.0. Audio is not included in this repository.

Ashraf et al., *Bolbosh: Script-Aware Flow Matching for Kashmiri Text-to-Speech*, arXiv:2603.07513.