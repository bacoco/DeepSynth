import pytest

np = pytest.importorskip("numpy")

from deepsynth.rag.storage import StateRef, StateShardReader, StateShardWriter


def test_state_shard_roundtrip(tmp_path):
    writer = StateShardWriter(tmp_path, max_shard_size_bytes=64)
    arr1 = np.arange(12, dtype=np.float32).reshape(3, 4)
    arr2 = np.ones((2, 4), dtype=np.float32)

    ref1 = writer.write(arr1)
    ref2 = writer.write(arr2)
    writer.close()

    reader = StateShardReader(tmp_path)
    out1 = reader.read(ref1)
    out2 = reader.read(ref2)

    assert np.allclose(out1, arr1.astype(np.float16))
    assert np.allclose(out2, arr2.astype(np.float16))

    manifest = writer.manifest()
    assert len(manifest) >= 1
    assert manifest[0]["entries"] >= 2
