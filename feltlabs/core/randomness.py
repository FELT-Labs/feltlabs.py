"""Module for generating random numbers used by models."""
import random

import numpy as np
from numpy.typing import NDArray


def set_seed(seed: int) -> None:
    """Set seed of random generator."""
    np.random.seed(seed)
    random.seed(seed)


def random_array_copy(array: NDArray, min: float, max: float) -> NDArray:
    """Create random array with same shape and data type as original array.

    Args:
        array: original array based on witch we want to create new array
        min: minumal value in new array
        max: maximum value in new array

    Returns:
        New random array of same shape and type as original array
    """
    return (max - min) * np.random.random(array.shape) + min
