"""Utility to generate summaries for evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.infer import DeepSeekSummarizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summaries for evaluation")
    parser.add_argument("input", help="JSONL file with records containing 'text'")
    parser.add_argument("--model", default="./deepseek-summarizer")
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()

    summarizer = DeepSeekSummarizer(args.model)

    records = []
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            summary = summarizer.summarize_text(
                record["text"],
                max_length=args.max_length,
                temperature=args.temperature,
            )
            handle.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
