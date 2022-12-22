"""Module providing model class interface."""
import copy
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from feltlabs.core import randomness
from feltlabs.core.json_handler import json_dump
from feltlabs.typing import PathType


class BaseModel(ABC):
    """Base model class for federated learning.
    Each model must subclass BaseModel so that necessary functions for federated
    learning are implemented.
    """

    model_type: str
    model_name: str
    sample_size: List[int] = [0]
    is_dirty: bool  # True if randomness was added

    @abstractmethod
    def __init__(self, data: dict):
        """Initialize model class from data dictionary.

        Args
            data: model loaded from JSON as dict
        """

    def export_model(self, filename: Optional[PathType] = None) -> bytes:
        """Export model to bytes and optionally store it as JSON file.

        Args:
            filename: path to exported file

        Returns:
            bytes of JSON file
        """
        data = self._export_data()
        data_bytes = json_dump(data)
        if filename:
            with open(filename, "wb") as f:
                f.write(data_bytes)

        return data_bytes

    def _set_seed(self, seed: int, param: str) -> None:
        """Set randomness seed of given parameter.

        Args:
            seed: seed provided for the whole training
            param: parameter for which we are setting the seed
        """
        seed = int(
            hashlib.sha256(bytes(f"{seed};{param}", "utf-8")).hexdigest(), 16
        ) % (2**32 - 1)
        randomness.set_seed(seed)

    # TODO: Specify typing for fit and predict
    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """Wrapper around model fitting.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shape (n_samples,)
        """

    @abstractmethod
    def aggregate(self, models: List["BaseModel"]) -> None:
        """Wrapper around aggregation function

        Args:
            models: list of models
        """

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediction of shape (n_samples, n_features)
        """

    @abstractmethod
    def add_noise(self, seed: int) -> None:
        """Add pseudo random noise to the model.

        Args:
            seed: randomness seed to generate pseudo random model
        """

    @abstractmethod
    def remove_noise_models(self, seeds: List[int]) -> None:
        """Remove generate and remove random models from current model based on seeds.

        Args:
            seeds: list of seeds used for generating random models
        """

    @abstractmethod
    def _export_data(self) -> dict:
        """Get model data as dictionary for storing (and loading) model.

        Returns:
            dictionary containing model data which should be stored
        """


class AvgModel(BaseModel):
    """Model based on Based Model necessary for basic averaging federated learning.
    Most of the models should subclass from this model.
    """

    ops = {
        "sum_op": lambda x, _: np.sum(x, axis=0),
        "mean_op": lambda x, _: np.mean(x, axis=0),
    }

    def new_model(self, params: Dict[str, NDArray] = {}) -> "AvgModel":
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
        assert len(self.sample_size) == 1, "Can't add randomness to aggregated model."
        rand_model = self.get_random_models([seed])
        self._agg_models_op(self.ops["sum_op"], rand_model)
        self.is_dirty = True

    def get_random_models(
        self, seeds: List[int], _min: int = -100, _max: int = 100
    ) -> List["AvgModel"]:
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
        # TODO: Right now we are not using "size" for the sklearn models
        for seed, size in zip(seeds, self.sample_size):
            params = self._get_params()
            new_params = {}
            for param, array in params.items():
                self._set_seed(seed, param)

                if type(array) == list:
                    value = [randomness.random_array_copy(a, _min, _max) for a in array]
                else:
                    value = randomness.random_array_copy(array, _min, _max)

                new_params[param] = value

            models.append(self.new_model(new_params))
        return models

    # TODO: Specify typing for fit and predict
    def fit(self, X: Any, y: Any) -> None:
        """Wrapper around model fitting.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shape (n_samples,)
        """
        self.sample_size = [len(y)]
        self._fit(X, y)

    def aggregate(self, models: List["AvgModel"]) -> None:
        """Wrapper around aggregation function

        Args:
            models: list of models
        """
        self.sample_size.extend([m.sample_size[0] for m in models])
        self._aggregate(models)

    def export_model(self, filename: Optional[PathType] = None) -> bytes:
        """Export model to bytes and optionally store it as JSON file.

        Args:
            filename: path to exported file

        Returns:
            bytes of JSON file
        """
        data = self._export_data()
        data_bytes = json_dump(data)
        if filename:
            with open(filename, "wb") as f:
                f.write(data_bytes)

        return data_bytes

    def _agg_models_op(self, op: Callable, models: List["AvgModel"]) -> None:
        """Perform aggregation operation on list of models.

        Args:
            op: function to run on on values, with definition fn(model_values, weights)
            models: list of models
        """
        models_params = [m._get_params() for m in [self, *models]]
        models_weights = np.array([m.sample_size[0] for m in [self, *models]])

        new_params = {}
        for param in models_params[0]:
            if not all(param in params for params in models_params):
                print(f"WARNING: Parameter '{param}' is not present in all models.")
                continue

            is_list = isinstance(models_params[0][param], list)

            values = [params[param] for params in models_params]
            # Convert to array of objects to prevent warning (elements have different shape)
            values = np.array(values, dtype=object) if is_list else values
            val = op(values, models_weights)

            if is_list:
                val = op(values, models_weights)
                new_params[param] = list(val)
            else:
                val = op(values, models_weights)
                new_params[param] = val

        self._set_params(new_params)

    @abstractmethod
    def _aggregate(self, models: List["AvgModel"]) -> None:
        """Aggregation function on self + list of models, specific for given model.

        Args:
            models: list of models
        """

    @abstractmethod
    def _fit(self, X: Any, y: Any) -> None:
        """Fit model on given data, specific for given model type.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shape (n_samples,)
        """

    @abstractmethod
    def _get_params(self) -> Dict[str, NDArray]:
        """Get dictionary of model parameters.

        Returns:
            dictionary of parameters as name to numpy array
        """

    @abstractmethod
    def _set_params(self, params: Dict[str, NDArray]) -> None:
        """Set values of model parameters.

        Args:
            params: dictionary mapping from name of param to numpy array
        """
