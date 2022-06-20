"""Module for performing federated averaging of models."""
import copy
from typing import Any

import numpy as np

from feltoken.typing import Model

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
        try:
            params[param] = getattr(model, param)
        except Exception:
            pass

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


def sum_models(models: list[Model]) -> Model:
    """Sum trainable parameters of scikit-learn models.

    Args:
        models: list of scikit-learn models.

    Returns:
        scikit-learn model with new values.
    """
    params = _get_models_params(models)
    new_params = {}
    for param, values in params.items():
        val = np.mean(values, axis=0)
        new_params[param] = val.astype(values[0].dtype)

    model = _set_model_params(copy.deepcopy(models[0]), new_params)
    return model


def random_model(model: Model, min=-100, max=100) -> Any:
    """Generate models with random parameters.

    Args:
        model: scikit-learn models.

    Returns:
        scikit-learn model with random values.
    """
    params = _get_model_params(model)
    new_params = {}
    for param, value in params.items():
        val = (max - min) * np.random.random(value.shape) + min
        new_params[param] = val.astype(val.dtype)

    model = _set_model_params(copy.deepcopy(model), new_params)
    return model
