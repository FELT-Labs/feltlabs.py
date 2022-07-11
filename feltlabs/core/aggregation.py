"""Module for performing federated averaging of models."""
import copy
import random
from typing import Any, Callable

import numpy as np

from feltlabs.typing import Model

ATTRIBUTE_LIST = ["coef_", "intercept_", "coefs_", "intercepts_"]


def _get_models_params(models: list[Model]) -> dict[str, list[np.ndarray]]:
    """Extract trainable parameters from scikit-learn models.

    Args:
        models: list of scikit-learn models.

    Returns:
        dictionary mapping attributes to list of values numpy arrays extracted from models.
    """
    params = {}
    for param in ATTRIBUTE_LIST:
        params[param] = []
        for model in models:
            if hasattr(model, param):
                params[param].append(getattr(model, param))
            else:
                params.pop(param, None)
                break

    return params


def _get_model_params(model: Model) -> dict[str, np.ndarray]:
    """Extract trainable parameters from scikit-learn model.

    Args:
        model: list of scikit-learn models.

    Returns:
        dictionary mapping attributes to list of values numpy arrays extracted from models.
    """
    params = {}
    for param in ATTRIBUTE_LIST:
        if hasattr(model, param):
            params[param] = getattr(model, param)

    return params


def _set_model_params(model: Model, params: dict[str, np.ndarray]) -> Model:
    """Set new values of trainable params to scikit-learn models.

    Args:
        model: scikit-learn model.
        params: dictinary mapping attributes to numpy arrays.

    Returns:
        scikit-learn model with new values.
    """
    for param, value in params.items():
        setattr(model, param, value)
    return model


def _agg_models_op(
    op: Callable,
    models: list[Model],
) -> Model:
    """Perform aggregation operation on list of scikit-learn models.

    Args:
        op: function to run on model values
        models: list of scikit-learn models

    Returns:
        scikit-learn model with new values based on op
    """
    params = _get_models_params(models)
    new_params = {}
    for param, values in params.items():
        val = op(values)
        new_params[param] = val.astype(values[0].dtype)

    model = _set_model_params(copy.deepcopy(models[0]), new_params)
    return model


def sum_models(models: list[Model]) -> Model:
    """Sum trainable parameters of scikit-learn models.

    Args:
        models: list of scikit-learn models.

    Returns:
        scikit-learn model with new values.
    """
    op = lambda x: np.sum(x, axis=0)
    return _agg_models_op(op, models)


def aggregate_models(models: list[Model]) -> Model:
    """Aggregate trainable parameters of scikit-learn models.

    Args:
        models: list of scikit-learn models.

    Returns:
        scikit-learn model with new values.
    """
    op = lambda x: np.mean(x, axis=0)
    return _agg_models_op(op, models)


def remove_noise_models(main_model: Model, random_models: list[Model]) -> Model:
    """Remove added noise of random models from main scikit-learn model.

    Args:
        model: main scikit-learn model
        models: list of scikit-learn models with random values.

    Returns:
        scikit-learn model with new values.
    """
    op = lambda x: -1 * np.mean(x, axis=0)
    noise_model = _agg_models_op(op, random_models)
    return sum_models([main_model, noise_model])


def _set_seed(seed: int) -> None:
    """Set seed of random generator."""
    np.random.seed(seed)
    random.seed(seed)


def random_model(model: Model, seed: int, min: int = -100, max: int = 100) -> Any:
    """Generate models with random parameters.

    Args:
        model: scikit-learn models.
        seed: seed for randomness generation
        min: minimum value of random number
        max: maximum value of random number

    Returns:
        scikit-learn model with random values.
    """
    _set_seed(seed)

    params = _get_model_params(model)
    new_params = {}
    for param, value in params.items():
        val = (max - min) * np.random.random(value.shape) + min
        new_params[param] = val.astype(val.dtype)

    model = _set_model_params(copy.deepcopy(model), new_params)
    return model
