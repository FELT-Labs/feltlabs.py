"""Testing model aggregation functions."""
import numpy as np

from feltlabs.core.models import sklearn_model

X = [[0], [1], [2]]
y = [[0, 1, 2], [1, 2, 3], [0, 2, 4]]


def test_sklearn_linreg_aggregation():
    model_data = {"model_type": "sklearn", "model_name": "LinearRegression"}

    model1 = sklearn_model.Model(model_data)
    model1.fit(X, y[0])

    model2 = sklearn_model.Model(model_data)
    model2.fit(X, y[1])

    model3 = sklearn_model.Model(model_data)
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), y[0])
    assert np.allclose(model2.predict(X), y[1])

    agg_model = model1.new_model()
    agg_model.aggregate([model2])
    assert np.allclose(agg_model.predict(X), [0.5, 1.5, 2.5])

    agg_model = model1.new_model()
    agg_model.aggregate([model3])
    assert np.allclose(agg_model.predict(X), [0, 1.5, 3])

    enc_model1 = model1.new_model()
    enc_model1.add_noise(10)
    assert not np.allclose(enc_model1.predict(X), y[0])

    enc_model2 = model2.new_model()
    enc_model2.add_noise(11)
    assert not np.allclose(enc_model2.predict(X), y[1])

    final_model = enc_model1
    final_model.aggregate([enc_model2])
    final_model.remove_noise_models([10, 11])
    assert np.allclose(final_model.predict(X), [0.5, 1.5, 2.5])


def test_sklearn_lasso_aggregation():
    model_data = {
        "model_type": "sklearn",
        "model_name": "Lasso",
        "init_params": {"alpha": 0.05},
    }

    model1 = sklearn_model.Model(model_data)
    model1.fit(X, y[0])

    model2 = sklearn_model.Model(model_data)
    model2.fit(X, y[1])

    model3 = sklearn_model.Model(model_data)
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), y[0], atol=0.20)
    assert np.allclose(model2.predict(X), y[1], atol=0.20)

    agg_model = model1.new_model()
    agg_model.aggregate([model2])
    assert np.allclose(agg_model.predict(X), [0.5, 1.5, 2.5], atol=0.20)

    agg_model = model1.new_model()
    agg_model.aggregate([model3])
    assert np.allclose(agg_model.predict(X), [0, 1.5, 3], atol=0.20)

    enc_model1 = model1.new_model()
    enc_model1.add_noise(10)
    assert not np.allclose(enc_model1.predict(X), y[0], atol=0.20)

    enc_model2 = model2.new_model()
    enc_model2.add_noise(11)
    assert not np.allclose(enc_model2.predict(X), y[1], atol=0.20)

    final_model = enc_model1
    final_model.aggregate([enc_model2])
    final_model.remove_noise_models([10, 11])
    assert np.allclose(final_model.predict(X), [0.5, 1.5, 2.5], atol=0.20)


def test_sklearn_ridge_aggregation():
    model_data = {
        "model_type": "sklearn",
        "model_name": "Ridge",
        "init_params": {"alpha": 0.05},
    }

    model1 = sklearn_model.Model(model_data)
    model1.fit(X, y[0])

    model2 = sklearn_model.Model(model_data)
    model2.fit(X, y[1])

    model3 = sklearn_model.Model(model_data)
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), y[0], atol=0.20)
    assert np.allclose(model2.predict(X), y[1], atol=0.20)

    agg_model = model1.new_model()
    agg_model.aggregate([model2])
    assert np.allclose(agg_model.predict(X), [0.5, 1.5, 2.5], atol=0.20)

    agg_model = model1.new_model()
    agg_model.aggregate([model3])
    assert np.allclose(agg_model.predict(X), [0, 1.5, 3], atol=0.20)

    enc_model1 = model1.new_model()
    enc_model1.add_noise(10)
    assert not np.allclose(enc_model1.predict(X), y[0], atol=0.20)

    enc_model2 = model2.new_model()
    enc_model2.add_noise(11)
    assert not np.allclose(enc_model2.predict(X), y[1], atol=0.20)

    final_model = enc_model1
    final_model.aggregate([enc_model2])
    final_model.remove_noise_models([10, 11])
    assert np.allclose(final_model.predict(X), [0.5, 1.5, 2.5], atol=0.20)
