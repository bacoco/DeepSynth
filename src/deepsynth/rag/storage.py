"""Disk-backed storage for encoder states."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class StateRef:
    """Pointer to an encoder state persisted on disk."""

    shard_id: str
    offset: int
    length: int
    shape: Tuple[int, ...]
    dtype: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "offset": self.offset,
            "length": self.length,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "StateRef":
        return cls(
            shard_id=str(data["shard_id"]),
            offset=int(data["offset"]),
            length=int(data["length"]),
            shape=tuple(int(x) for x in data["shape"]),
            dtype=str(data["dtype"]),
        )


class StateShardWriter:
    """Append-only writer that shards encoder states into binary blobs."""

    def __init__(
        self,
        directory: Path | str,
        *,
        max_shard_size_bytes: int = 256 * 1024 * 1024,
        dtype: str = "float16",
    ) -> None:
        self.base_dir = Path(directory)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_size_bytes = max_shard_size_bytes
        self.dtype = np.dtype(dtype)

        self._current_shard_id = -1
        self._current_file = None
        self._current_meta = None
        self._current_size = 0
        self._current_offset = 0
        self._manifest: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    def write(self, state: np.ndarray) -> StateRef:
        if self._current_file is None:
            self._open_new_shard()

        array = np.asarray(state, dtype=self.dtype)
        data = array.tobytes(order="C")
        if self._current_size + len(data) > self.max_shard_size_bytes:
            self._open_new_shard()

        offset = self._current_offset
        self._current_file.write(data)
        self._current_file.flush()
        self._current_size += len(data)
        self._current_offset += len(data)

        ref = StateRef(
            shard_id=self._shard_name(self._current_shard_id),
            offset=offset,
            length=array.size,
            shape=array.shape,
            dtype=str(array.dtype),
        )
        self._append_manifest(ref)
        return ref

    def close(self) -> None:
        if self._current_file:
            self._current_file.close()
            self._current_file = None
        if self._current_meta:
            self._current_meta.close()
            self._current_meta = None

    def manifest(self) -> List[Dict[str, object]]:
        return list(self._manifest.values())

    # ------------------------------------------------------------------
    def _open_new_shard(self) -> None:
        self.close()
        self._current_shard_id += 1
        shard_name = self._shard_name(self._current_shard_id)
        shard_path = self.base_dir / f"{shard_name}.bin"
        meta_path = self.base_dir / f"{shard_name}.manifest.jsonl"
        self._current_file = open(shard_path, "ab")
        self._current_meta = open(meta_path, "a", encoding="utf-8")
        self._current_size = os.path.getsize(shard_path)
        self._current_offset = self._current_size
        self._manifest[shard_name] = {
            "shard_id": shard_name,
            "path": str(shard_path),
            "manifest": str(meta_path),
            "size_bytes": self._current_size,
            "entries": 0,
        }

    def _append_manifest(self, ref: StateRef) -> None:
        shard = self._manifest[ref.shard_id]
        shard["size_bytes"] = self._current_size
        shard["entries"] = int(shard["entries"]) + 1
        assert self._current_meta is not None
        self._current_meta.write(json.dumps(ref.to_dict()) + "\n")
        self._current_meta.flush()

    def _shard_name(self, shard_id: int) -> str:
        return f"shard_{shard_id:05d}"

    # Context manager helpers -------------------------------------------------
    def __enter__(self) -> "StateShardWriter":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        self.close()


class StateShardReader:
    """Random-access reader for encoder states."""

    def __init__(self, directory: Path | str) -> None:
        self.base_dir = Path(directory)

    def read(self, ref: StateRef) -> np.ndarray:
        shard_path = self.base_dir / f"{ref.shard_id}.bin"
        dtype = np.dtype(ref.dtype)
        with open(shard_path, "rb") as handle:
            handle.seek(ref.offset)
            buffer = handle.read(ref.length * dtype.itemsize)
        array = np.frombuffer(buffer, dtype=dtype, count=ref.length)
        return array.reshape(ref.shape)

    def iter_manifest(self, shard_id: str) -> Iterable[StateRef]:
        meta_path = self.base_dir / f"{shard_id}.manifest.jsonl"
        if not meta_path.exists():
            return []
        with open(meta_path, "r", encoding="utf-8") as handle:
            for line in handle:
                yield StateRef.from_dict(json.loads(line))


__all__ = ["StateRef", "StateShardReader", "StateShardWriter"]
