"""Module for training models."""
from typing import Any

import numpy as np

from feltlabs.typing import Model


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
