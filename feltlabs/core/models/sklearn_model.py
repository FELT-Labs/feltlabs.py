"""Module for importing/exporting sklearn models to json."""
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn import linear_model, neighbors, neural_network, preprocessing

from feltlabs.core import randomness
from feltlabs.core.json_handler import json_dump
from feltlabs.typing import BaseModel, PathType

# TODO: we need to set warm_start=True for multiple rounds!!!
#     - maybe need to reset _no_improvement_count and max_iter (n_iter_)

# TODO: SVM attributes  ["dual_coef_", "support_", "support_vectors_", "_n_support"
# Attributes and data type casting for them (done only after removing randomness)
ATTRIBUTE_LIST = {
    "coef_": None,
    "intercept_": None,
    "coefs_": None,
    "intercepts_": None,
    "classes_": lambda x: np.rint(x).astype(np.uint),
    "centroids_": None,
    "n_iter_": np.rint,
    "n_layers_": round,
    "n_outputs_": round,
    # Needed by MLPClassifier
    "t_": round,
}
FIXED_ATTRIBUTE_LIST = [
    "out_activation_",
    "loss_curve_",
    "best_loss_",
    "_no_improvement_count",
]

SUPPORTED_MODELS = {
    # Regression
    "LinearRegression": linear_model.LinearRegression,
    "Lasso": linear_model.Lasso,
    "Ridge": linear_model.Ridge,
    "ElasticNet": linear_model.ElasticNet,
    "LassoLars": linear_model.LassoLars,
    # Classification
    "LogisticRegression": linear_model.LogisticRegression,
    "SGDClassifier": linear_model.SGDClassifier,
    # Clustering
    "NearestCentroidClassifier": neighbors.NearestCentroid,
    # Neural Networks
    # TODO: Limit size of hidden layers
    "MLPClassifier": neural_network.MLPClassifier,
    "MLPRegressor": neural_network.MLPRegressor,
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
        self.is_dirty = data.get("is_dirty", False)

        model_class = SUPPORTED_MODELS[self.model_name]
        self.model = model_class(**data.get("init_params", {}))

        params = data.get("model_params", {})
        self._set_params(params)

        self.sample_size = data.get("sample_size", self.sample_size)

        if self.is_dirty:
            # Substract random models (generated from seeds) from loaded model
            self.remove_noise_models(data.get("seeds", []))
        else:
            self._init_post_clean()

    def _init_post_clean(self):
        """Run extra initialization for clean model."""
        # Bit hacky solution adding label binarizer for MLPClassifier
        if (
            self.model_name == "MLPClassifier"
            and not hasattr(self.model, "_label_binarizer")
            and hasattr(self.model, "classes_")
        ):
            self.model._label_binarizer = preprocessing.LabelBinarizer()
            self.model._label_binarizer.fit(self.model.classes_)

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
            "is_dirty": self.is_dirty,
            "init_params": self.model.get_params(),  # Get params of sklearn models
            "model_params": {
                **self._get_params(),
                **self._get_params(FIXED_ATTRIBUTE_LIST),
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

                if type(array) == NDArray:
                    value = randomness.random_array_copy(array, _min, _max)
                elif type(array) == list:
                    value = [randomness.random_array_copy(a, _min, _max) for a in array]
                else:
                    value = randomness.random_array_copy(np.array([array]), _min, _max)[
                        0
                    ]

                new_params[param] = value

            models.append(self.new_model(new_params))
        return models

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
        self._set_params(self._get_params(), type_cast=True)
        # Update sample size, because now we have clean aggregated model
        self.sample_size = [sum(self.sample_size)]
        self.is_dirty = False
        # Init other params based on clean model
        self._init_post_clean()

    def _get_params(
        self, attributes: list = list(ATTRIBUTE_LIST)
    ) -> dict[str, NDArray]:
        """Get dictionary of model parameters.

        Args:
            attributes: list of attributes to get from the object

        Returns:
            dictionary of parameters as name to numpy array
        """
        params = {}
        for p in attributes:
            if hasattr(self.model, p) and getattr(self.model, p) is not None:
                params[p] = getattr(self.model, p)
        return params

    def _set_params(self, params: dict[str, NDArray], type_cast: bool = False) -> None:
        """Set values of model parameters.

        Args:
            params: dictionary mapping from name of param to numpy array
            type_cast: set true if types should be cast to expected type
        """
        for param, value in params.items():
            if type_cast and ATTRIBUTE_LIST[param] is not None:
                value = ATTRIBUTE_LIST[param](value)
            setattr(self.model, param, value)

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
        self.model.fit(X, y)

    def predict(self, X: Any) -> Any:
        """Use mode for prediction on given data.

        Args:
            X: array like data used for prediciton of shape (n_samples, n_features)
        """
        return self.model.predict(X)
