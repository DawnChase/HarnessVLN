from __future__ import annotations

import numpy as np

from harness.media import FileArrayStore, decode_media_refs


def test_array_store_round_trips_nested_arrays_and_cleans_up(tmp_path) -> None:
    store = FileArrayStore(tmp_path)
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    encoded = store.encode({"rgb": source, "metadata": ["fixture"]})
    root = store.root
    decoded = decode_media_refs(encoded)

    assert root is not None and root.is_dir()
    assert isinstance(decoded["rgb"], np.memmap)
    np.testing.assert_array_equal(decoded["rgb"], source)
    assert decoded["metadata"] == ["fixture"]

    del decoded
    store.close()
    assert not root.exists()
