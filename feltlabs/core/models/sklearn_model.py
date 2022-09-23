"""Module for importing/exporting sklearn models to json."""
import json
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn import linear_model

from feltlabs.core import randomness
from feltlabs.typing import BaseModel, PathType

ATTRIBUTE_LIST = ["coef_", "intercept_", "coefs_", "intercepts_", "classes_", "n_iter_"]
SUPPORTED_MODELS = {
    "LogisticRegression": linear_model.LogisticRegression,
    "LinearRegression": linear_model.LinearRegression,
}


class Model(BaseModel):
    """Model class for scikit-learn models implementing BaseModel."""

    model_type: str = "sklearn"

    def __init__(self, data: dict):
        """Initialize model calss from data dictionary.

        Args
            data: model loaded from JSON as dict
        """
        if data["model_name"] not in SUPPORTED_MODELS:
            raise Exception("Unsupported model type")

        self.model_name = data["model_name"]

        model_class = SUPPORTED_MODELS[self.model_name]
        self.model = model_class(**data.get("init_params", {}))

        params = {p: np.array(v) for p, v in data.get("model_params", {}).items()}
        self._set_params(params)

        # Substract random models (generated from seeds) from loaded model
        if "seeds" in data:
            self.remove_noise_models(data.get("seeds", []))

    def export_model(self, filename: Optional[PathType] = None) -> bytes:
        """Export sklean model to JSON file or return it as bytes.

        Args:
            filename: path to exported file

        Returns:
            bytes of JSON file
        """
        data = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "init_params": self.model.get_params(),  # Get params of sklearn models
            "model_params": self._get_params(to_list=True),
        }

        if filename:
            with open(filename, "w") as f:
                json.dump(data, f)

        return bytes(json.dumps(data), "utf-8")

    def get_random_model(self, seed: int, min: int = -100, max: int = 100) -> BaseModel:
        """Generate models with random parameters.

        Args:
            seed: seed for randomness generation
            min: minimum value of random number
            max: maximum value of random number

        Returns:
            Returns copy of model with random variables.
        """
        randomness.set_seed(seed)

        params = self._get_params()
        new_params = {}
        for param, array in params.items():
            new_params[param] = randomness.random_array_copy(array, min, max)

        model = self.new_model(new_params)
        return model

    def remove_noise_models(self, seeds: list[int]) -> None:
        """Remove generate and remove random models from current model based on seeds.

        Args:
            seeds: list of seeds used for generating random models
        """
        if len(seeds) == 0:
            return

        noise_models = [self.get_random_model(s) for s in seeds]

        op = lambda x: -1 * np.mean(x, axis=0)

        n_model, *other_models = noise_models
        n_model._agg_models_op(op, other_models)

        self._agg_models_op(self.ops["sum_op"], [n_model])

    def _get_params(self, to_list: bool = False) -> dict[str, NDArray]:
        """Get dictionary of model parameters.

        Args:
            to_list: flag to convert numpy arrays to lists (used for export)

        Returns:
            dictionary of parameters as name to numpy array
        """
        params = {}
        for p in ATTRIBUTE_LIST:
            if hasattr(self.model, p):
                params[p] = (
                    getattr(self.model, p).tolist()
                    if to_list
                    else getattr(self.model, p)
                )
        return params

    def _set_params(self, params: dict[str, NDArray]) -> None:
        """Set values of model parameters.

        Args:
            params: dictionary mapping from name of param to numpy array
        """
        for param, value in params.items():
            setattr(self.model, param, value)

    def aggregate(self, models: list[BaseModel]) -> None:
        """Aggregation function on self + list of models.

        Args:
            models: list of models
        """
        if len(models) == 0:
            return
        self._agg_models_op(self.ops["mean_op"], models)

    def fit(self, X: Any, y: Any) -> None:
        """Fit model on given data.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shapre (n_samples,)
        """
        self.model.fit(X, y)

    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediciton of shape (n_samples, n_features)
        """
        return self.model.predict(X)
