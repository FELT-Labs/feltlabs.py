"""Module for loading data and models."""
from typing import Any

import numpy as np

from feltlabs.config import AggregationConfig, TrainingConfig
from feltlabs.core.cryptography import decrypt_nacl
from feltlabs.core.ocean import get_dataset_files
from feltlabs.core.storage import load_model


# TODO: Add model type
def load_data(config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load data and return them in (X, y) format."""
    if config.data_type == "test":
        # Demo data for testing
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        return X, y
    else:
        files = get_dataset_files(config)
        if config.data_type == "csv":
            X, y = [], []
            for f in files:
                data = np.genfromtxt(f, delimiter=",")
                X.append(data[:-1])
                y.append(data[-1])

            return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    raise Exception("No data loaded.")


def load_models(config: AggregationConfig) -> list[Any]:
    """Load models for aggregation."""
    files = get_dataset_files(config)
    # Decrypt models using private key
    models = []
    for file_path in files:
        with open(file_path, "rb") as f:
            data = decrypt_nacl(config.private_key, f.read())
            models.append(load_model(data))

    return models
