"""Model for performing federated averaging of models."""
from typing import Any

import numpy as np

ATTRIBUTE_LIST = ["coef_", "intercept_", "coefs_", "intercepts_"]


def _get_models_params(models: list[Any]) -> dict[str, list[np.ndarray]]:
    """Extract trainable parameters from scikit-learn models.

    Args:
        modesl: list of scikit-learn models.

    Returns:
        dictionary mapping attributes to list of values numpy arrays extracted from models.
    """
    params = {}
    for param in ATTRIBUTE_LIST:
        params[param] = []
        try:
            for model in models:
                params[param].append(getattr(model, param))
        except Exception:
            params.pop(param, None)

    return params


def _set_model_params(model: Any, params: dict[str, np.ndarray]):
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


def average_models(models: list[Any]) -> Any:
    """Average trainable parameters of scikit-learn models.

    Args:
        models: list of scikit-learn models.

    Returns:
        scikit-learn model with new values.
    """
    params = _get_models_params(models)
    average_params = {}
    for param, values in params.items():
        val = np.mean(values, axis=0)
        average_params[param] = val.astype(values[0].dtype)

    model = _set_model_params(models[0], average_params)
    return model
