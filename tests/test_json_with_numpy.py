"""Test saving and loading JSON files with numpy arrays."""
from pathlib import Path

import numpy as np
from deepdiff import DeepDiff

from feltlabs.core.json_handler import json_dump, json_load

obj = {
    "params": np.random.random((4, 5)),
    "params2": [
        np.ones((2, 5), dtype=np.uint),
        np.ones((3, 5), dtype=np.float32),
        np.random.random((4, 5)),
    ],
    "model_type": {"p1": np.ones((2,))},
    "model_name": "LinearRegression",
}


def test_json_numpy_encoding(tmp_path: Path):
    file = tmp_path / "out.json"

    with file.open("wb") as f:
        f.write(json_dump(obj))

    obj_loaded = json_load(file)
    ## Test for no difference between loaded and original
    assert {} == DeepDiff(obj_loaded, obj)
