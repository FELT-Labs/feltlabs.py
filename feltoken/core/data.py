import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from feltoken.algorithm.config import TrainingConfig

INPUT_FOLDER = Path("/data/inputs/")


def get_files() -> list[Path]:
    """Get all files provided in Ocean's compute job environment."""
    files = []
    dids = json.loads(os.getenv("DIDS", "[]"))
    for did in dids:
        # In future we might need to do different actions based on DID
        # just list all files in DID folder for now
        files.extend(list(INPUT_FOLDER.joinpath(did).glob("*")))
    return files


# TODO: Add model type
def load_data(config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load data and return them in (X, y) format."""
    if config.data_type == "test":
        # Demo data for testing
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        return X, y
    else:
        files = get_files()
        if config.data_type == "csv":
            X, y = [], []
            for f in files:
                data = np.genfromtxt(f, delimiter=",")
                X.append(data[:-1])
                y.append(data[-1])

            return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    raise Exception("No data loaded.")


def load_models(config: TrainingConfig) -> list[Any]:
    """Load models for aggregation."""
    files = get_files()
    return files
