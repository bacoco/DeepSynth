"""Command-line helpers for the multi-vector RAG pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from deepsynth.rag.index import MultiVectorIndex
from deepsynth.rag.pipeline import ManifestChunk, PipelineManifest
from deepsynth.rag.storage import StateShardReader, StateShardWriter


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if "chunks" in data:
        return data["chunks"]
    raise ValueError("Manifest must be a list or contain a 'chunks' key")


def cmd_build_index(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)
    index_dir = output_dir / "index"
    state_dir = output_dir / "states"
    index_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(manifest_path)
    writer = StateShardWriter(state_dir, max_shard_size_bytes=args.max_shard_bytes)
    index = MultiVectorIndex(dim=args.dim, default_agg=args.agg)
    chunks: List[ManifestChunk] = []

    for entry in manifest:
        vectors_path = Path(entry["vectors_path"])
        state_path = Path(entry["state_path"])
        doc_id = entry["doc_id"]
        chunk_id = entry["chunk_id"]
        metadata = entry.get("metadata", {})

        vectors = np.load(vectors_path)
        state = np.load(state_path)
        state_ref = writer.write(state)
        index.add_chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            search_vectors=vectors,
            state_ref=state_ref,
            metadata=metadata,
        )
        chunks.append(
            ManifestChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                state_ref=state_ref.to_dict(),
                metadata=metadata,
            )
        )

    index.save(index_dir)
    writer.close()

    pipeline_manifest = PipelineManifest(
        index={
            "dim": index.dim,
            "default_agg": index.default_agg,
            "total_vectors": index.total_vectors,
            "total_chunks": index.total_chunks,
        },
        storage=writer.manifest(),
        chunks=chunks,
    )
    (output_dir / "pipeline_manifest.json").write_text(
        json.dumps(pipeline_manifest.to_dict(), indent=2)
    )
    print(json.dumps(pipeline_manifest.to_dict(), indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    index_dir = Path(args.index)
    state_dir = Path(args.states)
    query_vector = np.load(args.query)
    index = MultiVectorIndex.load(index_dir)
    reader = StateShardReader(state_dir)

    results = index.search(query_vector, top_k=args.top_k, agg=args.agg)
    output = []
    for result in results:
        state = reader.read(result.state_ref)
        output.append(
            {
                "doc_id": result.doc_id,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "vector_scores": result.vector_scores,
                "state_shape": list(state.shape),
                "metadata": result.metadata,
            }
        )
    print(json.dumps(output, indent=2))


def cmd_inspect(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text())
    summary = {
        "index": data.get("index", {}),
        "storage_shards": len(data.get("storage", [])),
        "chunks": len(data.get("chunks", [])),
    }
    if args.verbose:
        summary["chunks_detail"] = data.get("chunks", [])
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSynth multi-vector RAG tools")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index", help="Build an index from a manifest of vectors/states")
    build.add_argument("--manifest", required=True, help="Path to manifest (JSON or JSONL)")
    build.add_argument("--output", required=True, help="Directory to write index + shards")
    build.add_argument("--dim", type=int, default=4096, help="Vector dimension")
    build.add_argument("--agg", choices=["max", "sum"], default="max")
    build.add_argument(
        "--max-shard-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Maximum size per shard in bytes",
    )
    build.set_defaults(func=cmd_build_index)

    query = sub.add_parser("query", help="Search an existing index using a precomputed query vector")
    query.add_argument("--index", required=True, help="Path to saved index directory")
    query.add_argument("--states", required=True, help="Path to state shards directory")
    query.add_argument("--query", required=True, help="Path to .npy file with query vector")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--agg", choices=["max", "sum"], default="max")
    query.set_defaults(func=cmd_query)

    inspect_cmd = sub.add_parser("inspect", help="Inspect a pipeline manifest")
    inspect_cmd.add_argument("--manifest", required=True, help="Path to pipeline_manifest.json")
    inspect_cmd.add_argument("--verbose", action="store_true", help="Include chunk details")
    inspect_cmd.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
