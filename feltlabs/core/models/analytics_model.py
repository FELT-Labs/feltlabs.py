"""Module for handling analytics models."""
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union

import numpy as np
from numpy.typing import NDArray

from feltlabs.core import randomness
from feltlabs.core.models.base_model import AvgModel, BaseModel


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


class SingleModel(AvgModel):
    """Model class for calculating single statistic implementing BaseModel."""

    model_type: str = "analytics"
    model: Dict[str, NDArray] = {
        "value": np.array([0]),
    }

    def __init__(self, data: dict):
        """Initialize model class from data dictionary.

        Args
            data: model loaded from JSON as dict
        """
        if data["model_name"] not in SUPPORTED_MODELS:
            raise Exception("Unsupported model type")

        self.model_name = data["model_name"]
        self.metric = SUPPORTED_MODELS[data["model_name"]]
        self.is_dirty = data.get("is_dirty", False)

        params = data.get("model_params", {})
        self._set_params(params)

        self.sample_size = data.get("sample_size", self.sample_size)
        if self.is_dirty:
            # Subtract random models (generated from seeds) from loaded model
            self.remove_noise_models(data.get("seeds", []))

    def _export_data(self) -> dict:
        """Get model data as dictionary for storing (and loading) model.

        Returns:
            dictionary containing model data which should be stored
        """
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "is_dirty": self.is_dirty,
            "model_params": self._get_params(),
            "sample_size": self.sample_size,
        }

    def get_random_models(
        self, seeds: List[int], _min: int = -100, _max: int = 100
    ) -> List[AvgModel]:
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
        ), f"Can't generate random models. Num seeds ({len(seeds)}) and sizes ({len(self.sample_size)}) mismatch."

        models = []
        for seed, size in zip(seeds, self.sample_size):
            params = self._get_params()
            new_params = {}
            for param, array in params.items():
                self._set_seed(seed, param)

                new_params[param] = randomness.random_array_copy(array, _min, _max)
                if self.metric.scale_rand:
                    new_params[param] *= max(1, size)

            models.append(self.new_model(new_params))
        return models

    def remove_noise_models(self, seeds: List[int]) -> None:
        """Remove generate and remove random models from current model based on seeds.

        Args:
            seeds: list of seeds used for generating random models
        """
        if len(seeds) == 0 or not self.is_dirty:
            return

        noise_models = self.get_random_models(seeds)

        n_model, *other_models = noise_models
        n_model._agg_models_op(self.metric.remove_fn, other_models)

        self._agg_models_op(self.ops["sum_op"], [n_model])
        # Update sample size, because now we have clean aggregated model
        self.sample_size = [sum(self.sample_size)]
        self.is_dirty = False

    def _get_params(self) -> Dict[str, NDArray]:
        """Get dictionary of model parameters.

        Returns:
            dictionary of parameters as name to numpy array
        """
        return self.model

    def _set_params(self, params: Dict[str, NDArray]) -> None:
        """Set values of model parameters.

        Args:
            params: dictionary mapping from name of param to numpy array
        """
        self.model = {**self.model, **params}

    def _aggregate(self, models: List[AvgModel]) -> None:
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
            y: array like target values of shape (n_samples,)
        """
        self.model["value"] = self.metric.fit_fn(y)

    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediction of shape (n_samples, n_features)
        """
        print(
            f"{self.model_name} value is {self.metric.output_fn(self.model['value'], self.sample_size)}"
        )
        return self.metric.output_fn(self.model["value"], self.sample_size)


class Model(BaseModel):
    """Model class for single or multiple analytics."""

    model_type: str = "analytics"
    models = []

    def __init__(self, data: Union[dict, list]):
        """Initialize model class from data dictionary.

        Args
            data: model loaded from JSON as dict
        """
        data = data if isinstance(data, list) else [data]
        self.models = [SingleModel(d) for d in data]

    def _export_data(self) -> Union[dict, list]:
        """Get model data as dictionary for storing (and loading) model.

        Returns:
            dictionary containing model data which should be stored
        """
        if len(self.models) == 1:
            return self.models[0]._export_data()
        return [model._export_data() for model in self.models]

    def add_noise(self, seed: int) -> None:
        """Add pseudo random noise to the model.

        Args:
            seed: randomness seed to generate pseudo random model
        """
        # TODO: increment seed for each model (seed + i), need extra remove
        for model in self.models:
            model.add_noise(seed)

    def remove_noise_models(self, seeds: List[int]) -> None:
        """Remove generate and remove random models from current model based on seeds.

        Args:
            seeds: list of seeds used for generating random models
        """
        for model in self.models:
            model.remove_noise_models(seeds)

    def aggregate(self, models: List["Model"]) -> None:
        """Wrapper around aggregation function

        Args:
            models: list of models
        """
        for model, *other_models in zip(self.models, *[m.models for m in models]):
            model.aggregate(other_models)

    def fit(self, X: Any, y: Any) -> None:
        """Fit model on given data.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shape (n_samples,)
        """
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediction of shape (n_samples, n_features)
        """
        if len(self.models) == 1:
            return self.models[0].predict(X)
        return [model.predict(X) for model in self.models]
