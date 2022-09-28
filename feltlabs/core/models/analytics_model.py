"""Module for importing/exporting analytics models to json."""
import json
from ast import Call
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from feltlabs.core import randomness
from feltlabs.typing import BaseModel, PathType


@dataclass
class Metric:
    """Class for defining different analytics which can be calculated.

    Attributes:
        scale_rand: set true if random values are scaled by sample size
        fit_fn: function for calculating metric on original data
        agg_fun: aggregation function for combining results from different models
        remove_fn: function used for calculate negative of aggregated noise model
        output_fn: function used to calculate final value
    """

    scale_rand: bool
    fit_fn: Callable
    agg_fn: Callable
    remove_fn: Callable
    output_fn: Callable = lambda x, _: x


SUPPORTED_MODELS = {
    "Sum": Metric(
        scale_rand=True,
        fit_fn=lambda x: np.sum(x, axis=0),
        agg_fn=lambda vals, _: np.sum(vals, axis=0),
        remove_fn=lambda rands, _: -np.sum(rands, axis=0),
    ),
    "Mean": Metric(
        scale_rand=False,
        fit_fn=lambda x: np.mean(x, axis=0),
        agg_fn=lambda vals, weights: (weights.T @ vals) / np.sum(weights, axis=0),
        remove_fn=lambda rands, weights: -(weights.T @ rands) / np.sum(weights, axis=0),
    ),
    "Variance": Metric(
        # Calculating variance using sums should be numerically more stable
        scale_rand=False,
        fit_fn=lambda x: np.array(
            [
                np.sum(np.power(x, 2), axis=0),
                np.sum(x, axis=0),
            ]
        ),
        agg_fn=lambda vals, _: np.sum(vals, axis=0),
        remove_fn=lambda rands, _: -np.sum(rands, axis=0),
        output_fn=lambda vals, size: abs(
            (vals[0] - vals[1] ** 2 / sum(size)) / sum(size)
        ),
    ),
    "Std": Metric(
        # Calculating variance using sums should be numerically more stable
        scale_rand=False,
        fit_fn=lambda x: np.array(
            [
                np.sum(np.power(x, 2), axis=0),
                np.sum(x, axis=0),
            ]
        ),
        agg_fn=lambda vals, _: np.sum(vals, axis=0),
        remove_fn=lambda rands, _: -np.sum(rands, axis=0),
        output_fn=lambda vals, size: np.sqrt(
            np.abs((vals[0] - vals[1] ** 2 / sum(size)) / sum(size))
        ),
    ),
}


class Model(BaseModel):
    """Model class for scikit-learn models implementing BaseModel."""

    model_type: str = "analytics"
    model: dict[str, NDArray] = {
        "value": np.array([0]),
    }

    def __init__(self, data: dict):
        """Initialize model calss from data dictionary.

        Args
            data: model loaded from JSON as dict
        """
        if data["model_name"] not in SUPPORTED_MODELS:
            raise Exception("Unsupported model type")

        self.model_name = data["model_name"]
        self.metric = SUPPORTED_MODELS[data["model_name"]]

        params = {p: np.array(v) for p, v in data.get("model_params", {}).items()}
        self._set_params(params)

        self.sample_size = data.get("sample_size", self.sample_size)
        # Substract random models (generated from seeds) from loaded model
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
            "model_params": self._get_params(to_list=True),
            "sample_size": self.sample_size,
        }

        if filename:
            with open(filename, "w") as f:
                json.dump(data, f)

        return bytes(json.dumps(data), "utf-8")

    def get_random_models(
        self, seeds: list[int], _min: int = -100, _max: int = 100
    ) -> list[BaseModel]:
        """Generate models with random parameters.

        Args:
            seeds: list of seeds for randomness generation
            _min: minimum value of random number
            _max: maximum value of random number

        Returns:
            Returns copy of model with random variables.
        """
        assert len(seeds) == len(
            self.sample_size
        ), f"Can't generate random models. Num seeds ({len(seeds)}) and sizes ({len(self.sample_size)}) missmatch."

        models = []
        for seed, size in zip(seeds, self.sample_size):
            randomness.set_seed(seed)

            params = self._get_params()
            new_params = {}
            for param, array in params.items():
                new_params[param] = randomness.random_array_copy(array, _min, _max)
                if self.metric.scale_rand:
                    new_params[param] *= max(1, size)

            models.append(self.new_model(new_params))
        return models

    def remove_noise_models(self, seeds: list[int]) -> None:
        """Remove generate and remove random models from current model based on seeds.

        Args:
            seeds: list of seeds used for generating random models
        """
        if len(seeds) == 0:
            return

        noise_models = self.get_random_models(seeds)

        n_model, *other_models = noise_models
        n_model._agg_models_op(self.metric.remove_fn, other_models)

        self._agg_models_op(self.ops["sum_op"], [n_model])
        # Update sample size, because now we have clean aggregated model
        self.sample_size = [sum(self.sample_size)]

    def _get_params(self, to_list: bool = False) -> dict[str, NDArray]:
        """Get dictionary of model parameters.

        Args:
            to_list: flag to convert numpy arrays to lists (used for export)

        Returns:
            dictionary of parameters as name to numpy array
        """
        if to_list:
            return {k: v.tolist() for k, v in self.model.items()}
        return self.model

    def _set_params(self, params: dict[str, NDArray]) -> None:
        """Set values of model parameters.

        Args:
            params: dictionary mapping from name of param to numpy array
        """
        self.model = {**self.model, **params}

    def _aggregate(self, models: list[BaseModel]) -> None:
        """Aggregation function on self + list of models.

        Args:
            models: list of models
        """
        if len(models) == 0:
            return
        self._agg_models_op(self.metric.agg_fn, models)

    def _fit(self, X: Any, y: Any) -> None:
        """Fit model on given data.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shapre (n_samples,)
        """
        self.model["value"] = self.metric.fit_fn(y)

    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediciton of shape (n_samples, n_features)
        """
        print(
            f"{self.model_name} value is {self.metric.output_fn(self.model['value'], self.sample_size)}"
        )
        return self.metric.output_fn(self.model["value"], self.sample_size)
