"""Downloads Wav2Vec2 model, runs inference on LibriSpeech samples, saves predictions.csv"""

import csv
import time
import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import io
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

MODEL_ID = "facebook/wav2vec2-base-960h"
NUM_SAMPLES = 30
OUTPUT_PATH = "results/predictions.csv"


def load_model():
    print(f"Loading model: {MODEL_ID}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def load_data():
    REPO_ID = "hf-internal-testing/librispeech_asr_dummy"
    print("Fetching dataset file list...")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    parquet_file = parquet_files[0]

    print(f"Downloading {parquet_file}...")
    local_path = hf_hub_download(repo_id=REPO_ID, filename=parquet_file, repo_type="dataset")

    df = pd.read_parquet(local_path)
    samples = []
    for _, row in df.head(NUM_SAMPLES).iterrows():
        audio_bytes = row["audio"]["bytes"]
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        samples.append({
            "audio": {"array": audio_array, "sampling_rate": sr},
            "text": row["text"],
        })
    return samples


def run_inference(processor, model, samples):
    results = []
    for i, sample in enumerate(samples):
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        ground_truth = sample["text"].strip().upper()

        inputs = processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )

        start_time = time.time()
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        latency = time.time() - start_time

        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0].strip()

        results.append({
            "audio_id": f"{i:04d}",
            "ground_truth": ground_truth,
            "prediction": prediction,
            "latency_sec": round(latency, 4),
        })
        print(f"[{i+1}/{len(samples)}] GT: {ground_truth[:50]}... | Pred: {prediction[:50]}...")

    return results


def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_id", "ground_truth", "prediction", "latency_sec"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved predictions to {OUTPUT_PATH}")


def main():
    processor, model = load_model()
    samples = load_data()
    results = run_inference(processor, model, samples)
    save_results(results)


if __name__ == "__main__":
    main()
