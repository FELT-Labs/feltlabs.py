"""Module providing model class interface."""
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from feltlabs.typing import PathType


class BaseModel(ABC):
    """Base model class for federated learning.
    Each model must subclass BaseModel so that neccessary functions for federated
    learning are implemented.
    """

    model_type: str
    model_name: str

    ops = {
        "sum_op": lambda x: np.sum(x, axis=0),
        "mean_op": lambda x: np.mean(x, axis=0),
    }

    @abstractmethod
    def __init__(self, data: dict):
        """Initialize model calss from data dictionary.

        Args
            data: model loaded from JSON as dict
        """

    def new_model(self, params: dict[str, NDArray] = {}) -> "BaseModel":
        """Create copy of model and set new parameters.

        Args:
            params: values to set to the new model
        """
        new_model = copy.deepcopy(self)
        new_model._set_params(params)
        return new_model

    def add_noise(self, seed: int) -> None:
        """Add pseudo random noise to the model.

        Args:
            seed: randomness seed to generate pseudo random model
        """
        rand_model = self.get_random_model(seed)
        self._agg_models_op(self.ops["sum_op"], [rand_model])

    def _agg_models_op(self, op: Callable, models: list["BaseModel"]) -> None:
        """Perform aggregation operation on list of models.

        Args:
            op: function to run on model values
            models: list of models
        """
        if len(models) == 0:
            return

        models_params = [m._get_params() for m in [self, *models]]
        new_params = {}
        for param in models_params[0]:
            if not all(param in params for params in models_params):
                print(f"WARNING: Parameter '{param}' is not present in all models.")
                continue

            values = [params[param] for params in models_params]
            val = op(values)
            new_params[param] = val.astype(values[0].dtype)

        self._set_params(new_params)

    @abstractmethod
    def export_model(self, filename: Optional[PathType] = None) -> bytes:
        """Export model to bytes and optionaly store it as JSON file.

        Args:
            filename: path to exported file

        Returns:
            bytes of JSON file
        """

    @abstractmethod
    def get_random_model(
        self, seed: int, min: int = -100, max: int = 100
    ) -> "BaseModel":
        """Generate models with random parameters.

        Args:
            seed: seed for randomness generation
            min: minimum value of random number
            max: maximum value of random number

        Returns:
            Returns copy of model with random variables.
        """

    @abstractmethod
    def remove_noise_models(self, noise_models: list[int]) -> None:
        """Remove generate and remove random models from current model based on seeds.

        Args:
            seeds: list of seeds used for generating random models
        """

    @abstractmethod
    def _get_params(self, to_list: bool = False) -> dict[str, NDArray]:
        """Get dictionary of model parameters.

        Args:
            to_list: flag to convert numpy arrays to lists (used for export)

        Returns:
            dictionary of parameters as name to numpy array
        """

    @abstractmethod
    def _set_params(self, params: dict[str, NDArray]) -> None:
        """Set values of model parameters.

        Args:
            params: dictionary mapping from name of param to numpy array
        """

    @abstractmethod
    def aggregate(self, models: list["BaseModel"]) -> None:
        """Aggregation function on self + list of models.

        Args:
            models: list of models
        """

    # TODO: Specify typing for fit and predict
    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """Fit model on given data.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shapre (n_samples,)
        """

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediciton of shape (n_samples, n_features)
        """
