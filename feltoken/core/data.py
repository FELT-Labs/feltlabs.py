"""Module for loading data and models."""
from typing import Any

import numpy as np

from feltoken.algorithm.config import AggregationConfig, TrainingConfig
from feltoken.core.cryptography import decrypt_nacl
from feltoken.core.ocean import get_dataset_files
from feltoken.core.storage import bytes_to_model


# TODO: Add model type
def load_data(config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load data and return them in (X, y) format."""
    if config.data_type == "test":
        # Demo data for testing
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        return X, y
    else:
        files = get_dataset_files()
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
    files = get_dataset_files()
    # Decrypt models using private key
    models = []
    for file_path in files:
        with open(file_path, "rb") as f:
            data = decrypt_nacl(config.private_key, f.read())
            models.append(bytes_to_model(data))

    return models
