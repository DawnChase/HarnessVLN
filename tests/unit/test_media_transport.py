from __future__ import annotations

from pathlib import Path

import numpy as np

from harness.media import FileArrayStore, decode_media_refs


def test_array_store_round_trips_nested_arrays_and_cleans_up(tmp_path) -> None:
    store = FileArrayStore(tmp_path)
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    encoded = store.encode({"rgb": source, "metadata": ["fixture"]})
    root = store.root
    decoded = decode_media_refs(encoded)

    assert root is not None and root.is_dir()
    assert isinstance(decoded["rgb"], np.ndarray)
    assert not isinstance(decoded["rgb"], np.memmap)
    np.testing.assert_array_equal(decoded["rgb"], source)
    assert decoded["metadata"] == ["fixture"]

    del decoded
    store.close()
    assert not root.exists()


def test_array_store_releases_one_job_without_closing_the_store(tmp_path) -> None:
    store = FileArrayStore(tmp_path)
    first = store.encode(np.ones((2, 2), dtype=np.uint8), scope="job-a")
    second = store.encode(np.zeros((2, 2), dtype=np.uint8), scope="job-b")
    first_path = Path(first["$harness_array"]["path"])
    second_path = Path(second["$harness_array"]["path"])
    retained = decode_media_refs(first)

    store.release("job-a")

    assert not first_path.exists()
    assert second_path.exists()
    np.testing.assert_array_equal(retained, np.ones((2, 2), dtype=np.uint8))
    assert not isinstance(retained, np.memmap)
    assert store.root is not None and store.root.is_dir()
    store.close()
