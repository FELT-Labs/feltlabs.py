"""Module for loading data and models."""
import json
from typing import Any

import numpy as np
import requests

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
                X.append(data[:, :-1])
                y.append(data[:, -1])

            return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    raise Exception("No data loaded.")


def load_models(config: AggregationConfig) -> list[Any]:
    """Load models for aggregation.
    It either download models from urls or load them from input path.
    """
    data_array = []
    if config.download_models:
        with config.custom_data_path.open("r") as f:
            conf = json.load(f)

        for url in conf["model_urls"]:
            res = requests.get(url)
            data_array.append(res.content)

    else:
        files = get_dataset_files(config)
        # Decrypt models using private key
        for file_path in files:
            with open(file_path, "rb") as f:
                data_array.append(f.read())

    models = []
    for val in data_array:
        data = decrypt_nacl(config.private_key, val)
        models.append(load_model(data))

    return models
