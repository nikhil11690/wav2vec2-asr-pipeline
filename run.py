"""Entry point: runs inference then evaluation end-to-end."""

from src.inference import main as run_inference
from src.evaluate import main as run_evaluation

if __name__ == "__main__":
    print("=== Step 1: Running inference ===")
    run_inference()

    print("\n=== Step 2: Running evaluation ===")
    run_evaluation()

    print("\n✅ Done. Check the results/ folder for predictions.csv, metrics.json, report.md")
