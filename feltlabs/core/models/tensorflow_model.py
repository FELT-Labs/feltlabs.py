"""Module for importing/exporting tensorflow models to json."""
from typing import Any, Optional, Union, cast

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from feltlabs.core import randomness
from feltlabs.core.json_handler import json_dump, json_load
from feltlabs.typing import BaseModel, PathType

# TODO: Handle different array types

# So far include only models from tf.keras.applications
SUPPORTED_MODELS = {
    "MobileNetV2": tf.keras.applications.mobilenet_v2.MobileNetV2,
    "EfficientNetB0": tf.keras.applications.efficientnet.EfficientNetB0,
}


class Model(BaseModel):
    """Model class for tensorflow models implementing BaseModel."""

    model_type: str = "tensorflow"

    def __init__(self, data: dict):
        """Initialize model calss from data dictionary.

        Args
            data: model loaded from JSON as dict
        """
        if data["model_name"] not in SUPPORTED_MODELS:
            raise Exception("Unsupported model type")

        self.model_name = data["model_name"]
        self.is_dirty = data.get("is_dirty", False)

        model_class = SUPPORTED_MODELS[self.model_name]
        self.init_params = data.get("init_params", {})
        self.model = model_class(**self.init_params)

        params = data.get("model_params", {})
        self._set_params(params)

        self.sample_size = data.get("sample_size", self.sample_size)

        if self.is_dirty:
            # Substract random models (generated from seeds) from loaded model
            self.remove_noise_models(data.get("seeds", []))
        # Compile model
        self.model.compile(
            "adam", "sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def export_model(self, filename: Optional[PathType] = None) -> bytes:
        """Export model to JSON file or return it as bytes.

        Args:
            filename: path to exported file

        Returns:
            bytes of JSON file
        """
        data = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "is_dirty": self.is_dirty,
            "init_params": self.init_params,
            "model_params": {
                **self._get_params(),
            },
            "sample_size": self.sample_size,
        }

        data_bytes = json_dump(data)
        if filename:
            with open(filename, "wb") as f:
                f.write(data_bytes)

        return data_bytes

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
        # TODO: Right now we are not using "size" for the sklearn models
        for seed, size in zip(seeds, self.sample_size):
            params = self._get_params()
            new_params = {}
            for param, array in params.items():
                randomness.set_seed(hash(f"{seed};{param}") % (2**32 - 1))

                if type(array) == list:
                    value = [randomness.random_array_copy(a, _min, _max) for a in array]
                else:
                    value = randomness.random_array_copy(array, _min, _max)

                new_params[param] = value

            models.append(self.new_model(new_params))
        return models

    def new_model(self, params: dict[str, NDArray] = {}) -> "BaseModel":
        """Create copy of model and set new parameters.

        Args:
            params: values to set to the new model
        """
        new_model = Model(json_load(self.export_model()))
        new_model._set_params(params)
        return new_model

    def remove_noise_models(self, seeds: list[int]) -> None:
        """Remove generate and remove random models from current model based on seeds.

        Args:
            seeds: list of seeds used for generating random models
        """
        if len(seeds) == 0 or not self.is_dirty:
            return

        noise_models = self.get_random_models(seeds)

        op = lambda x, _: -1 * np.mean(x, axis=0)

        n_model, *other_models = noise_models
        n_model._agg_models_op(op, other_models)

        self._agg_models_op(self.ops["sum_op"], [n_model])
        # Cast parameters to required types
        self._set_params(self._get_params())
        # Update sample size, because now we have clean aggregated model
        self.sample_size = [sum(self.sample_size)]
        self.is_dirty = False

    def _get_params(self) -> dict[str, NDArray]:
        """Get dictionary of model parameters.

        Returns:
            dictionary of parameters as name to numpy array
        """
        weights = cast(list[NDArray], self.model.get_weights())
        return dict(map(lambda x: (str(x[0]), x[1]), enumerate(weights)))

    def _set_params(self, params: dict[str, NDArray]) -> None:
        """Set values of model parameters.

        Args:
            params: dictionary mapping from name of param to numpy array
            type_cast: set true if types should be cast to expected type
        """
        if not params:
            return
        sorted_keys = map(str, sorted(map(int, params.keys())))
        weights = [params[i] for i in sorted_keys]
        self.model.set_weights(weights)

    def _aggregate(self, models: list[BaseModel]) -> None:
        """Aggregation function on self + list of models.

        Args:
            models: list of models
        """
        if len(models) == 0:
            return
        self._agg_models_op(self.ops["mean_op"], models)

    def _fit(self, X: Any, y: Any) -> None:
        """Fit model on given data.

        Args:
            X: array like training data of shape (n_samples, n_features)
            y: array like target values of shapre (n_samples,)
        """
        self.model.fit(X, y, epochs=1, batch_size=32)

    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediciton of shape (n_samples, n_features)
        """
        return self.model.predict(X)
