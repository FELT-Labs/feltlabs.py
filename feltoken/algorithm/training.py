"""Module for executing model training."""
from typing import Any

import numpy as np

from feltoken.algorithm.config import parse_training_args
from feltoken.core.aggregation import random_model, sum_models
from feltoken.core.data import load_data
from feltoken.core.ocean import save_output
from feltoken.core.storage import (
    encrypt_model,
    export_model,
    load_model,
    model_to_bytes,
)
from feltoken.typing import Model


def train_model(
    model: Model,
    data: tuple[np.ndarray, np.ndarray],
) -> Any:
    """Universal model training function, it starts training depending on selected type.

    Args:
        model: initial model object with fit() method
        data: tuple[X, Y] or string depending on training type

    Returns:
        new model object with trained weights
    """
    model.fit(data[0], data[1])
    return model


def main():
    """Main function for exectuting from command line."""
    args = parse_training_args()
    # Load data and model
    model = load_model(args.model)
    data = load_data(args)
    # Train model
    model = train_model(model, data)
    # Add randomness to model and encrypt using public key for aggregation
    rand_model = random_model(model)
    model = sum_models([model, rand_model])

    enc_model = encrypt_model(model, args.aggregation_key)
    enc_rand_model = encrypt_model(rand_model, args.public_key)

    # Save models into output
    save_output("model", enc_model)
    save_output("rand_model", enc_rand_model)


if __name__ == "__main__":
    main()
