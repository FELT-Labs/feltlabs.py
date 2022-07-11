"""Testing model aggregation functions."""
import numpy as np
from sklearn.linear_model import LinearRegression

from feltlabs.core.aggregation import (
    aggregate_models,
    random_model,
    remove_noise_models,
    sum_models,
)

X = [[0], [1], [2]]
y = [[0, 1, 2], [1, 2, 3], [0, 2, 4]]


def test_aggregation():
    model1 = LinearRegression()
    model1.fit(X, y[0])

    model2 = LinearRegression()
    model2.fit(X, y[1])

    model3 = LinearRegression()
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), y[0])
    assert np.allclose(model2.predict(X), y[1])

    agg_model = aggregate_models([model1, model2])
    assert np.allclose(agg_model.predict(X), [0.5, 1.5, 2.5])

    agg_model = aggregate_models([model1, model3])
    assert np.allclose(agg_model.predict(X), [0, 1.5, 3])

    rand_model1 = random_model(model1, 10)
    enc_model1 = sum_models([model1, rand_model1])
    assert not np.allclose(enc_model1.predict(X), y[0])

    rand_model2 = random_model(model2, 11)
    enc_model2 = sum_models([model2, rand_model2])
    assert not np.allclose(enc_model2.predict(X), y[1])

    tmp_model = aggregate_models([enc_model1, enc_model2])
    final_model = remove_noise_models(tmp_model, [rand_model1, rand_model2])
    assert np.allclose(final_model.predict(X), [0.5, 1.5, 2.5])
