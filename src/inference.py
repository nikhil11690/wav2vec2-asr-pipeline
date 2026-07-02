"""Downloads Wav2Vec2 model, runs inference on LibriSpeech samples, saves predictions.csv"""

import csv
import time
import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset

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
    print("Loading LibriSpeech test set (streaming)...")
    dataset = load_dataset(
        "librispeech_asr", "clean", split="test.clean", streaming=True
    )
    samples = list(dataset.take(NUM_SAMPLES))
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
