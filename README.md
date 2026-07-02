# Speech Recognition Pipeline — Wav2Vec2

Reproducible ASR inference and evaluation pipeline using `facebook/wav2vec2-base-960h`.

## Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Run
```bash
python run.py
```
This downloads the model + dataset, runs inference on 30 LibriSpeech test samples, and writes results to `results/`.

## Outputs
- `results/predictions.csv` — per-sample ground truth vs prediction
- `results/metrics.json` — WER, CER, latency stats
- `results/report.md` — human-readable summary

## Model
`facebook/wav2vec2-base-960h` — self-supervised speech representation model fine-tuned for CTC-based ASR on 960 hours of LibriSpeech.

## Dataset
LibriSpeech `test.clean` split (streamed via Hugging Face `datasets`).
