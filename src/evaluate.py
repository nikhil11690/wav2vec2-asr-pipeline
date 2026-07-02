"""Computes WER, CER, latency stats from predictions.csv, saves metrics.json + report.md"""

import csv
import json
import statistics
import os
from jiwer import wer, cer

PREDICTIONS_PATH = "results/predictions.csv"
METRICS_PATH = "results/metrics.json"
REPORT_PATH = "results/report.md"


def load_predictions():
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def compute_metrics(rows):
    references = [r["ground_truth"] for r in rows]
    hypotheses = [r["prediction"] for r in rows]
    latencies = [float(r["latency_sec"]) for r in rows]

    metrics = {
        "model": "facebook/wav2vec2-base-960h",
        "num_samples": len(rows),
        "wer": round(wer(references, hypotheses), 4),
        "cer": round(cer(references, hypotheses), 4),
        "avg_latency_sec": round(statistics.mean(latencies), 4),
        "min_latency_sec": round(min(latencies), 4),
        "max_latency_sec": round(max(latencies), 4),
    }
    return metrics


def save_metrics(metrics):
    os.makedirs("results", exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


def write_report(metrics):
    report = f"""# Evaluation Report

## Model
{metrics['model']}

## Summary Statistics
| Metric | Value |
|---|---|
| Number of Samples | {metrics['num_samples']} |
| Word Error Rate (WER) | {metrics['wer']} |
| Character Error Rate (CER) | {metrics['cer']} |
| Avg Inference Latency (sec) | {metrics['avg_latency_sec']} |
| Min Latency (sec) | {metrics['min_latency_sec']} |
| Max Latency (sec) | {metrics['max_latency_sec']} |

## Interpretation
A WER of {metrics['wer']} means the model got approximately {round(metrics['wer']*100, 1)}% of words wrong on average across the evaluated samples. CER of {metrics['cer']} reflects character-level accuracy, which is typically lower since it's more forgiving of small spelling differences.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report to {REPORT_PATH}")


def main():
    rows = load_predictions()
    metrics = compute_metrics(rows)
    save_metrics(metrics)
    write_report(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
