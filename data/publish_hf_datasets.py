"""Utilities to normalise and publish datasets on the Hugging Face Hub.

This script reads the dataset references enumerated in the project
documentation and repackages them into a unified format compatible with the
training pipeline (``text``/``summary`` pairs, optionally with an ``image``
pointer).  Each processed dataset is uploaded to the Hub so it can be consumed
directly by the fine-tuning CLI without manual preprocessing.

The script is intentionally defensive: it provides rich logging, automatic
field detection and helpful error messages when a dataset does not expose the
expected schema.  For extremely large datasets you may limit the number of
records pushed via ``--max-records`` which samples each split deterministically.

💡 New in this revision: pass ``--generate-images`` to render PNG snapshots of
the textual documents directly within the publishing pipeline.  Images are
produced on the fly, batched in memory for the current upload only, and never
persisted to disk.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

from datasets import Dataset, DatasetDict, Image as ImageFeature, load_dataset

from .text_to_image import TextToImageConverter


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset specifications


TransformFn = Callable[[dict], Optional[dict]]


@dataclass
class FieldCandidates:
    """Heuristics used to discover relevant columns in a dataset."""

    text: Sequence[str]
    summary: Sequence[str]
    image: Sequence[str] = field(default_factory=tuple)


@dataclass
class PublishSpec:
    """Configuration describing how to normalise and publish a dataset."""

    source: str
    repo_suffix: str
    subset: Optional[str] = None
    splits: Sequence[str] = ("train", "validation", "test")
    candidates: FieldCandidates = field(default_factory=lambda: FieldCandidates((), ()))
    description: Optional[str] = None
    transform: Optional[TransformFn] = None

    def resolve_field(self, columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        """Return the first matching column in ``columns`` according to ``candidates``."""

        if not candidates:
            return None

        lookup = {column.lower(): column for column in columns}
        for candidate in candidates:
            if candidate in columns:
                return candidate
            lowered = candidate.lower()
            if lowered in lookup:
                return lookup[lowered]
        # Fallback to substring search
        for candidate in candidates:
            lowered = candidate.lower()
            for column in columns:
                if lowered in column.lower():
                    return column
        return None


# ---------------------------------------------------------------------------
# Normalisation helpers


def _coerce_to_text(value: object) -> str:
    """Convert a dataset field into a printable string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        # Flatten nested structures (e.g. list of answers) into a space joined string
        return " ".join(_coerce_to_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        # Preserve deterministic ordering for reproducibility
        return " ".join(_coerce_to_text(value[key]) for key in sorted(value))
    return str(value)


def _default_transform(row: dict, *, text_field: str, summary_field: str, image_field: Optional[str], source: str) -> Optional[dict]:
    text = _coerce_to_text(row.get(text_field))
    summary = _coerce_to_text(row.get(summary_field))
    if not text or not summary:
        return None

    record = {
        "text": text,
        "summary": summary,
        "source_dataset": source,
    }

    if image_field:
        record["image"] = row.get(image_field)
    return record


def _generate_images_on_the_fly(
    dataset: Dataset,
    *,
    converter: TextToImageConverter,
    batch_size: int,
) -> Dataset:
    """Attach an ``image`` column by rendering documents on the fly."""

    def _convert_batch(batch: dict) -> dict:
        texts = batch.get("text", [])
        images = [converter.convert(text) for text in texts]
        return {"image": images}

    augmented = dataset.map(
        _convert_batch,
        batched=True,
        batch_size=batch_size,
    )

    try:
        augmented = augmented.cast_column("image", ImageFeature())
    except Exception as exc:  # pragma: no cover - depends on datasets version
        LOGGER.warning("Failed to cast generated images to Image feature: %s", exc)

    return augmented


def _prepare_split(
    spec: PublishSpec,
    split: str,
    *,
    max_records: Optional[int] = None,
    generate_images: bool,
    converter: Optional[TextToImageConverter],
    image_batch_size: int,
) -> Optional[Dataset]:
    try:
        dataset = load_dataset(spec.source, spec.subset, split=split)
    except Exception as exc:  # pragma: no cover - network / auth errors
        LOGGER.warning("Skipping split %s of %s: %s", split, spec.source, exc)
        return None

    columns = dataset.column_names
    text_field = spec.resolve_field(columns, spec.candidates.text)
    summary_field = spec.resolve_field(columns, spec.candidates.summary)
    image_field = spec.resolve_field(columns, spec.candidates.image)

    if spec.transform is None and (text_field is None or summary_field is None):
        raise RuntimeError(
            f"Unable to infer text/summary fields for {spec.source}."
            f" Available columns: {columns}"
        )

    def transform_row(row: dict) -> dict:
        record = (
            spec.transform(row)
            if spec.transform is not None
            else _default_transform(
                row,
                text_field=text_field or "",
                summary_field=summary_field or "",
                image_field=image_field,
                source=spec.source,
            )
        )
        return record or {}

    processed = dataset.map(
        transform_row,
        remove_columns=columns,
    )
    processed = processed.filter(lambda row: bool(row.get("text") and row.get("summary")))

    if max_records is not None and len(processed) > max_records:
        processed = processed.select(range(max_records))

    if generate_images and "image" not in processed.column_names:
        if converter is None:
            converter = TextToImageConverter()
        LOGGER.info(
            "Generating images on the fly for %s/%s (batch size=%d)",
            spec.source,
            split,
            image_batch_size,
        )
        processed = _generate_images_on_the_fly(
            processed,
            converter=converter,
            batch_size=image_batch_size,
        )

    return processed


def build_dataset_dict(
    spec: PublishSpec,
    *,
    max_records: Optional[int] = None,
    generate_images: bool,
    converter: Optional[TextToImageConverter],
    image_batch_size: int,
) -> DatasetDict:
    """Create a :class:`DatasetDict` ready to be pushed to the Hub."""

    splits: Dict[str, Dataset] = {}
    for split in spec.splits:
        dataset = _prepare_split(
            spec,
            split,
            max_records=max_records,
            generate_images=generate_images,
            converter=converter,
            image_batch_size=image_batch_size,
        )
        if dataset is not None and len(dataset) > 0:
            splits[split] = dataset

    if not splits:
        raise RuntimeError(f"No valid splits generated for {spec.source}")

    return DatasetDict(splits)


# ---------------------------------------------------------------------------
# CLI


def publish(
    spec: PublishSpec,
    *,
    repo_prefix: str,
    token: Optional[str],
    private: bool,
    max_records: Optional[int],
    dry_run: bool,
    generate_images: bool,
    converter: Optional[TextToImageConverter],
    image_batch_size: int,
) -> None:
    repo_id = f"{repo_prefix}/{spec.repo_suffix}" if repo_prefix else spec.repo_suffix
    LOGGER.info("Preparing dataset %s → %s", spec.source, repo_id)
    dataset_dict = build_dataset_dict(
        spec,
        max_records=max_records,
        generate_images=generate_images,
        converter=converter,
        image_batch_size=image_batch_size,
    )

    if dry_run:
        for split, split_dataset in dataset_dict.items():
            LOGGER.info("Split %s: %s records", split, len(split_dataset))
        LOGGER.info("Dry run enabled - skipping push to %s", repo_id)
        return

    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message=f"Add dataset derived from {spec.source}",
    )
    LOGGER.info("Successfully pushed %s", repo_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish DeepSeek OCR training datasets to the Hugging Face Hub")
    parser.add_argument("--repo-prefix", default="deepseek-summaries", help="Organisation or user prefix for the Hub repo")
    parser.add_argument("--token", help="Hugging Face token (falls back to cached login)")
    parser.add_argument("--private", action="store_true", help="Create private repositories")
    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional maximum number of records to upload per split (useful for smoke tests)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare datasets without uploading them")
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help=(
            "Render missing image columns from the text content on the fly."
            " Images exist only during the upload batch and are not stored on disk."
        ),
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=64,
        help="Batch size to use when generating images on the fly",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Subset of dataset repo suffixes to publish (defaults to all)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_arg_parser()
    args = parser.parse_args()

    specs = list(PUBLISH_SPECS)
    if args.datasets:
        requested = set(args.datasets)
        specs = [spec for spec in specs if spec.repo_suffix in requested]
        missing = requested - {spec.repo_suffix for spec in specs}
        if missing:
            raise SystemExit(f"Unknown dataset specifiers: {sorted(missing)}")

    converter = TextToImageConverter() if args.generate_images else None

    for spec in specs:
        publish(
            spec,
            repo_prefix=args.repo_prefix,
            token=args.token,
            private=args.private,
            max_records=args.max_records,
            dry_run=args.dry_run,
            generate_images=args.generate_images,
            converter=converter,
            image_batch_size=args.image_batch_size,
        )


# ---------------------------------------------------------------------------
# Dataset specification registry (aligned with project documentation)


PUBLISH_SPECS: Sequence[PublishSpec] = [
    PublishSpec(
        source="ccdv/cnn_dailymail",
        subset="3.0.0",
        repo_suffix="cnn_dailymail",
        description="CNN / DailyMail articles paired with human written highlights.",
        candidates=FieldCandidates(text=("article",), summary=("highlights",)),
    ),
    PublishSpec(
        source="EdinburghNLP/xsum",
        repo_suffix="xsum",
        description="Extreme summarisation dataset with one-sentence abstracts.",
        candidates=FieldCandidates(text=("document",), summary=("summary",)),
    ),
    PublishSpec(
        source="ccdv/arxiv-summarization",
        repo_suffix="arxiv",
        description="arXiv articles paired with their abstracts.",
        candidates=FieldCandidates(text=("article",), summary=("abstract",)),
    ),
    PublishSpec(
        source="gigaword",
        subset="gigaword",
        repo_suffix="gigaword",
        description="Gigaword headline generation corpus.",
        candidates=FieldCandidates(text=("document",), summary=("summary",)),
    ),
    PublishSpec(
        source="trigaten/findsum",
        repo_suffix="findsum",
        description="Financial document summarisation corpus.",
        candidates=FieldCandidates(
            text=("article", "document", "context", "text"),
            summary=("summary", "highlights", "target"),
        ),
    ),
    PublishSpec(
        source="HuggingFaceM4/Docmatix",
        repo_suffix="docmatix",
        description="Docmatix instruction-following document VQA corpus.",
        candidates=FieldCandidates(
            text=("question", "prompt", "instruction"),
            summary=("answer", "response"),
            image=("image", "image_path", "image_url"),
        ),
    ),
    PublishSpec(
        source="docvqa",
        repo_suffix="docvqa",
        description="Document VQA dataset mapped to question-answer format.",
        candidates=FieldCandidates(
            text=("question",),
            summary=("answers", "answer"),
            image=("image", "image_path"),
        ),
    ),
    PublishSpec(
        source="AmazonScience/document-haystack",
        repo_suffix="document_haystack",
        description="Multi-page document benchmark from Amazon Science.",
        candidates=FieldCandidates(
            text=("question", "query", "context"),
            summary=("answer", "summary", "target"),
            image=("image", "page", "image_path"),
        ),
    ),
    PublishSpec(
        source="opendatalab/OmniDocBench",
        repo_suffix="omnidocbench",
        description="OmniDocBench heterogeneous document understanding benchmark.",
        candidates=FieldCandidates(
            text=("question", "instruction", "prompt", "context"),
            summary=("answer", "response", "target"),
            image=("image", "image_path", "image_url"),
        ),
    ),
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

