#!/usr/bin/env python3
"""
CLI for Instruction-Based Inference.

Supports:
- Single query inference
- Batch processing from JSONL files
- Multiple output formats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deepsynth.inference.instruction_engine import (
    InstructionEngine,
    GenerationParams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def run_single_query(args):
    """Run single query inference."""
    LOGGER.info("=" * 80)
    LOGGER.info("Single Query Inference")
    LOGGER.info("=" * 80)

    # Load engine
    engine = InstructionEngine(
        model_path=args.model_path,
        use_text_encoder=not args.no_text_encoder,
    )

    # Generation params
    params = GenerationParams(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )

    # Generate
    LOGGER.info("Document: %s", args.document[:100] + "...")
    LOGGER.info("Instruction: %s", args.instruction)

    result = engine.generate(
        document=args.document,
        instruction=args.instruction,
        params=params,
    )

    # Print result
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result.answer)
    print("=" * 80)
    print(f"Tokens: {result.tokens_generated} | Time: {result.inference_time_ms:.2f}ms")
    print("=" * 80)


def run_batch_processing(args):
    """Run batch processing from JSONL file."""
    LOGGER.info("=" * 80)
    LOGGER.info("Batch Processing")
    LOGGER.info("=" * 80)
    LOGGER.info("Input: %s", args.input_file)
    LOGGER.info("Output: %s", args.output_file)

    # Load engine
    engine = InstructionEngine(
        model_path=args.model_path,
        use_text_encoder=not args.no_text_encoder,
    )

    # Generation params
    params = GenerationParams(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )

    # Process file
    engine.generate_from_file(
        input_path=args.input_file,
        output_path=args.output_file,
        params=params,
    )

    LOGGER.info("✅ Batch processing complete!")
    LOGGER.info("Results written to: %s", args.output_file)


def run_interactive_mode(args):
    """Run interactive Q&A session."""
    LOGGER.info("=" * 80)
    LOGGER.info("Interactive Mode")
    LOGGER.info("=" * 80)
    LOGGER.info("Type 'exit' to quit")
    LOGGER.info("=" * 80)

    # Load engine
    engine = InstructionEngine(
        model_path=args.model_path,
        use_text_encoder=not args.no_text_encoder,
    )

    # Generation params
    params = GenerationParams(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )

    # Get document once
    print("\nEnter document (or path to file):")
    document_input = input("> ").strip()

    if Path(document_input).exists():
        with open(document_input, "r", encoding="utf-8") as f:
            document = f.read()
        print(f"✓ Loaded document from: {document_input}")
    else:
        document = document_input

    print("\n" + "=" * 80)
    print("Document loaded. Enter your questions/instructions:")
    print("=" * 80)

    # Interactive loop
    while True:
        try:
            instruction = input("\n> ").strip()

            if instruction.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not instruction:
                continue

            # Generate
            result = engine.generate(document, instruction, params)

            # Print answer
            print(f"\n{result.answer}")
            print(f"\n[{result.tokens_generated} tokens, {result.inference_time_ms:.0f}ms]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run instruction-based inference with DeepSynth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Single query
  python run_instruction_inference.py \\
      --model-path ./models/deepsynth-qa \\
      --document "AI has transformed healthcare..." \\
      --instruction "What has AI transformed?"

  # Batch processing
  python run_instruction_inference.py \\
      --model-path ./models/deepsynth-qa \\
      --input-file queries.jsonl \\
      --output-file answers.jsonl

  # Interactive mode
  python run_instruction_inference.py \\
      --model-path ./models/deepsynth-qa \\
      --interactive

Input file format (JSONL):
  {"document": "...", "instruction": "..."}
  {"document": "...", "instruction": "..."}

Output file format (JSONL):
  {"document": "...", "instruction": "...", "answer": "...", "inference_time_ms": 234}
        """,
    )

    # Model config
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--no-text-encoder",
        action="store_true",
        help="Disable text encoder (use vision-only mode)",
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--document",
        type=str,
        help="Source document text (for single query)",
    )
    input_group.add_argument(
        "--input-file",
        type=str,
        help="Input JSONL file (for batch processing)",
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive Q&A mode",
    )

    # Single query params
    parser.add_argument(
        "--instruction",
        type=str,
        help="Question or instruction (required for single query)",
    )

    # Batch processing params
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output JSONL file (required for batch processing)",
    )

    # Generation parameters
    parser.add_argument("--max-length", type=int, default=256, help="Max output length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for beam search")

    args = parser.parse_args()

    # Validate arguments
    if args.document and not args.instruction:
        parser.error("--instruction is required when using --document")

    if args.input_file and not args.output_file:
        parser.error("--output-file is required when using --input-file")

    # Run appropriate mode
    try:
        if args.document:
            run_single_query(args)
        elif args.input_file:
            run_batch_processing(args)
        elif args.interactive:
            run_interactive_mode(args)
    except Exception as e:
        LOGGER.error("Inference failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
