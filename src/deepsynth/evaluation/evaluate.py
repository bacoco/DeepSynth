"""CLI utility to evaluate summaries using ROUGE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import SummaryMetrics, evaluate_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated summaries")
    parser.add_argument("reference", help="JSONL file with reference summaries")
    parser.add_argument("predictions", help="JSONL file with generated summaries")
    parser.add_argument("--output", help="Optional JSON file to save the metrics")

    args = parser.parse_args()

    def load_pairs(path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    reference_lines = load_pairs(args.reference)
    prediction_lines = load_pairs(args.predictions)

    if len(reference_lines) != len(prediction_lines):
        raise ValueError("Number of references and predictions must match")

    pairs = []
    for ref_line, pred_line in zip(reference_lines, prediction_lines):
        ref_obj = json.loads(ref_line)
        pred_obj = json.loads(pred_line)
        pairs.append((ref_obj.get("summary", ""), pred_obj.get("summary", "")))

    metrics: SummaryMetrics = evaluate_pairs(pairs)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics.__dict__, indent=2))
    else:
        print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
