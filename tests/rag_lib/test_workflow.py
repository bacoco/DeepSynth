from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("torch")
import torch

pytest.importorskip("PIL")
from PIL import Image

from deepsynth.rag.text_query_encoder import QueryEncoder
from deepsynth.rag_lib.adapter import LoRAAdapterManager
from deepsynth.rag_lib.config import LibraryConfig
from deepsynth.rag_lib.workflow import (
    DocumentPayload,
    SourceType,
    build_pipeline,
    ingest_corpus,
    query_corpus,
)


def _make_temp_dir(tmp_path: Path) -> Path:
    storage = tmp_path / "storage"
    storage.mkdir()
    return storage


class DummyEncoderModel:
    def __call__(self, image: Any, return_dict: bool = True):
        hidden = torch.ones((1, 6, 4096), dtype=torch.float32)
        if return_dict:
            return {"last_hidden_state": hidden}
        return hidden


class DummyTextEncoder:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def eval(self):  # pragma: no cover - compatibility hook
        return self

    def to(self, device: torch.device):  # pragma: no cover - compatibility hook
        return self

    def encode(self, batch: List[str], max_length: int = 128):
        self.calls.append(batch)
        return torch.ones((len(batch), 4096), dtype=torch.float32)


class DummyDecoder:
    def __init__(self) -> None:
        self.model = DummyModel()
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, encoder_state: np.ndarray, metadata: Dict[str, Any], *, prompt_override: str | None = None) -> str:
        self.calls.append({"metadata": metadata, "prompt_override": prompt_override})
        return f"summary:{metadata.get('source_type', 'unknown')}"


class DummyModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.loaded = False

    def to(self, device: torch.device):  # pragma: no cover - pass-through
        self.device = device
        return self

    def eval(self):  # pragma: no cover - pass-through
        self.loaded = True
        return self


@pytest.fixture()
def context(tmp_path: Path):
    config = LibraryConfig(storage_dir=_make_temp_dir(tmp_path))
    text_encoder = DummyTextEncoder()
    query_encoder = QueryEncoder(model=text_encoder, normalize=False)
    decoder = DummyDecoder()
    ingestion_context = build_pipeline(
        encoder_model=DummyEncoderModel(),
        query_encoder=query_encoder,
        decoder=decoder,
        config=config,
    )
    return ingestion_context, decoder


def test_ingest_and_query_workflow(context) -> None:
    ingestion_context, decoder = context
    payloads = [
        DocumentPayload(doc_id="doc_text", source=SourceType.TEXT, data="hello world", metadata={"tag": "text"}),
        DocumentPayload(
            doc_id="doc_image",
            source=SourceType.IMAGE,
            data=Image.new("RGB", (32, 32), color="blue"),
            metadata={"tag": "image"},
        ),
    ]
    manifest = ingest_corpus(ingestion_context, payloads)
    assert manifest.chunks, "Expected chunks in manifest"

    answer = query_corpus(ingestion_context, "what is inside?", top_k=1)
    assert answer.chunks
    assert answer.chunks[0].summary.startswith("summary:")
    assert decoder.calls, "Decoder should have been invoked"


def test_query_with_lora_manager(context) -> None:
    ingestion_context, decoder = context
    payload = DocumentPayload(doc_id="doc_text", source=SourceType.TEXT, data="lora", metadata={})
    ingest_corpus(ingestion_context, [payload])

    base_model = decoder.model

    def loader(model: DummyModel, adapter_path: str) -> DummyModel:
        model.loaded = True
        return model

    manager = LoRAAdapterManager(
        base_model=base_model,
        adapter_path="adapter",
        loader=loader,
        device=torch.device("cpu"),
    )
    query_corpus(ingestion_context, "question", lora_manager=manager)
    assert base_model.loaded
