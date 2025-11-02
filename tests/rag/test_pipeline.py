from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from deepsynth.rag.encoder import EncoderFeaturizer
from deepsynth.rag.pipeline import IngestChunk, RAGPipeline
from deepsynth.rag.storage import StateShardWriter
from deepsynth.rag.index import MultiVectorIndex


class DummyEncoder:
    def __call__(self, image, return_dict=True):
        if isinstance(image, torch.Tensor):
            hidden = image
        else:
            hidden = torch.tensor(image, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden.unsqueeze(0))


class DummyQueryEncoder:
    def encode(self, texts):
        vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        return vec


def dummy_decoder(state, metadata, **kwargs):
    return f"summary-{metadata.get('chunk_id', 'unknown')}"


def test_pipeline_ingest_and_query(tmp_path):
    encoder = DummyEncoder()
    featurizer = EncoderFeaturizer(encoder_model=encoder, vectors_per_chunk=2, selection_policy="uniform")

    writer = StateShardWriter(tmp_path / "states", max_shard_size_bytes=1024)
    index = MultiVectorIndex(dim=4)
    pipeline = RAGPipeline(
        featurizer=featurizer,
        index=index,
        storage_writer=writer,
        query_encoder=DummyQueryEncoder(),
        decoder=dummy_decoder,
    )

    chunk_a = IngestChunk(
        doc_id="docA",
        chunk_id="c1",
        image=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        metadata={"chunk_id": "c1"},
    )
    chunk_b = IngestChunk(
        doc_id="docB",
        chunk_id="c2",
        image=torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0]],
            dtype=torch.float32,
        ),
        metadata={"chunk_id": "c2"},
    )

    manifest = pipeline.ingest_documents([chunk_a, chunk_b])
    assert manifest.index["total_chunks"] == 2

    answer = pipeline.answer_query("test question", top_k=2)
    assert answer.chunks
    assert answer.chunks[0].doc_id == "docA"
    assert answer.chunks[0].summary.startswith("summary")

    pipeline.close()
