"""Module for exporting sklearn models to json."""
import json
from typing import Any, Union

import numpy as np
from sklearn import linear_model

from feltoken.typing import FileType, PathType

ATTRIBUTE_LIST = ["coef_", "intercept_", "coefs_", "intercepts_", "classes_", "n_iter_"]
SUPPORTED_MODELS = {
    "LogisticRegression": linear_model.LogisticRegression,
    "LinearRegression": linear_model.LinearRegression,
}


def _model_name(model: Any) -> str:
    """Get name of model class."""
    for name, val in SUPPORTED_MODELS.items():
        if isinstance(model, val):
            return name
    raise Exception("Trying to export unsupported model")


def export_model(
    model: Any, filename: PathType = "", to_bytes: bool = False
) -> Union[None, bytes]:
    """Export sklean model to JSON file or return it as bytes.

    Args:
        model: sklearn model
        filename: path to exported file
        to_bytes: if true return bytes without exporting to file

    Returns:
        bytes of JSON file if to_bytes is set true. Else it returns None
    """
    data = {
        "model_name": _model_name(model),
        "init_params": model.get_params(),
        "model_params": {},
    }

    for p in ATTRIBUTE_LIST:
        if hasattr(model, p):
            data["model_params"][p] = getattr(model, p).tolist()

    if to_bytes:
        return bytes(json.dumps(data), "utf-8")

    with open(filename, "w") as f:
        json.dump(data, f)


def import_model(file: FileType) -> Any:
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
    model = model_class(**data["init_params"])
    for name, values in data["model_params"].items():
        setattr(model, name, np.array(values))

    return model
