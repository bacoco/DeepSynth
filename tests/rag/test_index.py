import pytest

np = pytest.importorskip("numpy")

from deepsynth.rag.index import MultiVectorIndex
from deepsynth.rag.storage import StateRef


def make_state_ref(idx: int) -> StateRef:
    return StateRef(
        shard_id=f"shard_{idx:05d}",
        offset=idx * 10,
        length=8,
        shape=(2, 4),
        dtype="float16",
    )


def test_index_search_max_and_sum(tmp_path):
    index = MultiVectorIndex(dim=4, default_agg="max")

    vectors_a = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0]], dtype=np.float32)
    vectors_b = np.array([[0, 1, 0, 0]], dtype=np.float32)

    index.add_chunk(
        doc_id="docA",
        chunk_id="chunk1",
        search_vectors=vectors_a,
        state_ref=make_state_ref(0),
        metadata={"label": "A"},
    )
    index.add_chunk(
        doc_id="docB",
        chunk_id="chunk2",
        search_vectors=vectors_b,
        state_ref=make_state_ref(1),
        metadata={"label": "B"},
    )

    query = np.array([1.0, 0.2, 0.0, 0.0], dtype=np.float32)

    results_max = index.search(query, top_k=2, agg="max")
    assert results_max[0].chunk_id == "chunk1"
    assert results_max[0].metadata["label"] == "A"

    results_sum = index.search(query, top_k=2, agg="sum")
    assert results_sum[0].chunk_id == "chunk1"
    assert results_sum[1].chunk_id == "chunk2"

    save_dir = tmp_path / "index"
    index.save(save_dir)
    loaded = MultiVectorIndex.load(save_dir)
    loaded_results = loaded.search(query, top_k=2)
    assert loaded_results[0].chunk_id == "chunk1"
