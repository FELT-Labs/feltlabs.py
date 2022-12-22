"""Module for loading data and models."""
import csv
import json
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import requests
from numpy.typing import NDArray

from feltlabs.config import AggregationConfig, TrainingConfig
from feltlabs.core.cryptography import decrypt_nacl
from feltlabs.core.ocean import get_datasets
from feltlabs.core.storage import load_model


def _has_csv_header(file: Path) -> bool:
    """Check given CSV file if it contains header."""
    with file.open() as f:
        lines = "".join(f.readline() for i in range(5))
        return csv.Sniffer().has_header(lines)


# TODO: Add model type
def load_data(config: TrainingConfig) -> Tuple[NDArray, NDArray]:
    """Load data and return them in (X, y) format."""
    if config.data_type == "test":
        # Demo data for testing
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        return X, y

    datasets = get_datasets(config)

    X, y = [], []
    for dataset in datasets.values():
        data_type = (
            config.data_type if dataset.data_type == "default" else dataset.data_type
        )

        for file, index in dataset.files:
            if data_type == "csv":
                # Check for header
                has_header = _has_csv_header(file)
                data = np.genfromtxt(file, delimiter=",", skip_header=has_header)
                # Get target column index (using modulo to get be positive index)
                index = config.target_column % data.shape[1]
                X.append(
                    np.concatenate([data[:, :index], data[:, index + 1 :]], axis=1)
                )
                y.append(data[:, index])

            elif data_type == "pickle":
                with file.open("rb") as f:
                    data = pickle.load(f)

                # TODO: Do something with the index (probably tabular/tuple data)
                X.append(data[0])
                y.append(data[-1])

    assert len(y) > 0, "No data loaded"

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


def load_models(config: AggregationConfig) -> List[Any]:
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
        # Models passed as dataset
        for dataset in get_datasets(config).values():
            for file_path, _ in dataset.files:
                with open(file_path, "rb") as f:
                    data_array.append(f.read())

    models = []
    for val in data_array:
        data = decrypt_nacl(config.private_key, val)
        models.append(load_model(data))

    return models
