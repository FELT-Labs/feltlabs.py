"""Module for importing/exporting tensorflow models to json."""
import os
from typing import Any, Dict, List, cast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from feltlabs.core.json_handler import json_load
from feltlabs.core.models.base_model import AvgModel
from feltlabs.core.models.tensorflow.cnn_network import get_cnn_network

# TODO: Handle different array types

# So far include only models from tf.keras.applications
SUPPORTED_MODELS = {
    # TODO: Seems to be some issue with mobile net maybe batch norm layer?
    "MobileNetV2": tf.keras.applications.mobilenet_v2.MobileNetV2,
    "EfficientNetB0": tf.keras.applications.efficientnet.EfficientNetB0,
    "CustomCNN": get_cnn_network,
}


class Model(AvgModel):
    """Model class for tensorflow models implementing BaseModel."""

    # TODO: Document model parameters

    model_type: str = "tensorflow"

    def __init__(self, data: dict):
        """Initialize model class from data dictionary.

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

        self.fit_args = data.get("fit_args", {})
        self.sample_size = data.get("sample_size", self.sample_size)

        if self.is_dirty:
            # Subtract random models (generated from seeds) from loaded model
            self.remove_noise_models(data.get("seeds", []))
        # Compile model
        self.model.compile(
            "adam", "sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def _export_data(self) -> dict:
        """Get model data as dictionary for storing (and loading) model.

        Returns:
            dictionary containing model data which should be stored
        """
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "is_dirty": self.is_dirty,
            "init_params": self.init_params,
            "fit_args": self.fit_args,
            "model_params": {
                **self._get_params(),
            },
            "sample_size": self.sample_size,
        }

    def new_model(self, params: Dict[str, NDArray] = {}) -> "AvgModel":
        """Create copy of model and set new parameters.

        Args:
            params: values to set to the new model
        """
        new_model = Model(json_load(self.export_model()))
        new_model._set_params(params)
        return new_model

    def remove_noise_models(self, seeds: List[int]) -> None:
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

    def _get_params(self) -> Dict[str, NDArray]:
        """Get dictionary of model parameters.

        Returns:
            dictionary of parameters as name to numpy array
        """
        weights = cast(List[NDArray], self.model.get_weights())
        return dict(map(lambda x: (str(x[0]), x[1]), enumerate(weights)))

    def _set_params(self, params: Dict[str, NDArray]) -> None:
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

    def _aggregate(self, models: List[AvgModel]) -> None:
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
            y: array like target values of shape (n_samples,)
        """
        self.model.fit(X, y, **self.fit_args)

    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediction of shape (n_samples, n_features)
        """
        return self.model.predict(X)
