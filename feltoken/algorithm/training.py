"""Module for executing model training."""
from pathlib import Path
from typing import Any

import numpy as np

from feltoken.algorithm.config import TrainingConfig, parse_training_args
from feltoken.core.data import load_data
from feltoken.core.storage import encrypt_model, export_model, load_model
from feltoken.typing import Model

OUTPUT_FOLDER = Path("/data/outputs")

# TODO: type for the model


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

    model = load_model(args.model)
    data = load_data(args)

    model = train_model(model, data)
    encrypted_model = encrypt_model(model, args.public_key)
    with open(OUTPUT_FOLDER / "model", "wb+") as f:
        f.write(encrypted_model)


if __name__ == "__main__":
    main()
