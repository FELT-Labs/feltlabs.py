from typing import Union

import numpy as np
from sklearn import datasets

from feltoken.node.config import Config


def load_data(config: Config) -> Union[tuple, str]:
    """Load data and return them in (X, y) formta."""
    if config.data == "test":
        # Demo data for testing
        X, y = datasets.load_diabetes(return_X_y=True)
        subset = np.random.choice(X.shape[0], 100, replace=False)
        return X[subset], y[subset]
    else:
        try:
            data = np.genfromtxt(config.data, delimiter=",")
            return data[:-1], data[-1]
        except Exception as e:
            raise Exception(f"Unable to load {config.data}\n{e}")
