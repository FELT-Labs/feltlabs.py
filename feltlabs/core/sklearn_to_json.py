"""Module for exporting sklearn models to json."""
import json
from typing import Any, Optional, Union

import numpy as np
from sklearn import linear_model

from feltlabs.core.aggregation import random_model, remove_noise_models
from feltlabs.typing import FileType, Model, PathType

ATTRIBUTE_LIST = ["coef_", "intercept_", "coefs_", "intercepts_", "classes_", "n_iter_"]
SUPPORTED_MODELS = {
    "LogisticRegression": linear_model.LogisticRegression,
    "LinearRegression": linear_model.LinearRegression,
}


def _model_name(model: Model) -> str:
    """Get name of model class."""
    for name, val in SUPPORTED_MODELS.items():
        if isinstance(model, val):
            return name
    raise Exception("Trying to export unsupported model")


def export_model(model: Model, filename: Optional[PathType] = None) -> bytes:
    """Export sklean model to JSON file or return it as bytes.

    Args:
        model: sklearn model
        filename: path to exported file

    Returns:
        bytes of JSON file
    """
    data = {
        "model_name": _model_name(model),
        "init_params": model.get_params(),
        "model_params": {},
    }

    for p in ATTRIBUTE_LIST:
        if hasattr(model, p):
            data["model_params"][p] = getattr(model, p).tolist()

    if filename:
        with open(filename, "w") as f:
            json.dump(data, f)

    return bytes(json.dumps(data), "utf-8")


def import_model(file: FileType) -> Model:
    """Import sklearn model from file.

    Args
        file: path to model file or bytes

    Returns:
        sklearn model object
    """
    if type(file) is bytes:
        data = json.loads(file)
    else:
        with open(file, "r") as f:
            data = json.load(f)

    if data["model_name"] not in SUPPORTED_MODELS:
        raise Exception("Unsupported model type")

    model_class = SUPPORTED_MODELS[data["model_name"]]
    model = model_class(**data.get("init_params", {}))
    for name, values in data.get("model_params", {}).items():
        setattr(model, name, np.array(values))

    # Substract random models (generated from seeds) from loaded model
    if "seeds" in data:
        rand_models = [random_model(model, s) for s in data.get("seeds", [])]
        model = remove_noise_models(model, rand_models)

    return model
