# Speech Recognition Pipeline — Wav2Vec2

A reproducible ASR (Automatic Speech Recognition) inference and evaluation pipeline built using `facebook/wav2vec2-base-960h` from Hugging Face. This project downloads a pretrained speech model, runs inference on real audio samples, and evaluates transcription accuracy using standard ASR metrics.

## Overview

- **Model**: [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)
- **Dataset**: [`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) (LibriSpeech clean subset)
- **Task**: End-to-end speech-to-text inference and evaluation
- **Samples evaluated**: 30

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Run

```bash
python run.py
```

This single command:
1. Downloads the pretrained Wav2Vec2 model and processor from Hugging Face
2. Downloads LibriSpeech audio samples
3. Runs inference on each audio clip
4. Saves predictions to `results/predictions.csv`
5. Computes WER, CER, and latency metrics
6. Saves `results/metrics.json` and `results/report.md`

## Results

| Metric | Value |
|---|---|
| Number of Samples | 30 |
| Word Error Rate (WER) | 0.0474 |
| Character Error Rate (CER) | 0.0156 |
| Avg Inference Latency | 1.871 sec |
| Min Latency | 0.3746 sec |
| Max Latency | 8.2264 sec |

**Interpretation**: A WER of 0.0474 means the model transcribed approximately 95.3% of words correctly across the evaluated samples — consistent with published benchmarks for Wav2Vec2-base on clean LibriSpeech audio. The lower CER (0.0156) reflects that most errors were minor (e.g., single characters), not complete word-level failures.

## Project Structure