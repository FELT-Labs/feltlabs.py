"""Testing model aggregation functions."""
import numpy as np

from feltlabs.core.models import analytics_model

X = [[0], [1], [2]]
y = [[0, 1, 2], [1, 2, 3], [0, 2, 4]]


def test_analytics_sum_aggregation():
    model_data = {"model_type": "analytics", "model_name": "Sum"}

    model1 = analytics_model.SingleModel(model_data)
    model1.fit(X, y[0])

    model2 = analytics_model.SingleModel(model_data)
    model2.fit(X, y[1])

    model3 = analytics_model.SingleModel(model_data)
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), sum(y[0]))
    assert np.allclose(model2.predict(X), sum(y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model2])
    assert np.allclose(agg_model.predict(X), sum(y[0] + y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model3])
    assert np.allclose(agg_model.predict(X), sum(y[0] + y[2]))

    enc_model1 = model1.new_model()
    enc_model1.add_noise(10)
    assert not np.allclose(enc_model1.predict(X), sum(y[0]))

    enc_model2 = model2.new_model()
    enc_model2.add_noise(11)
    assert not np.allclose(enc_model2.predict(X), sum(y[1]))

    enc_model3 = model3.new_model()
    enc_model3.add_noise(12)
    assert not np.allclose(enc_model3.predict(X), sum(y[2]))

    final_model = enc_model1
    final_model.aggregate([enc_model2, enc_model3])
    final_model.remove_noise_models([10, 11, 12])
    assert np.allclose(final_model.predict(X), sum(y[0] + y[1] + y[2]))


def test_analytics_mean_aggregation():
    model_data = {"model_type": "analytics", "model_name": "Mean"}

    model1 = analytics_model.SingleModel(model_data)
    model1.fit(X, y[0])

    model2 = analytics_model.SingleModel(model_data)
    model2.fit(X, y[1])

    model3 = analytics_model.SingleModel(model_data)
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), np.mean(y[0]))
    assert np.allclose(model2.predict(X), np.mean(y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model2])
    assert np.allclose(agg_model.predict(X), np.mean(y[0] + y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model3])
    assert np.allclose(agg_model.predict(X), np.mean(y[0] + y[2]))

    enc_model1 = model1.new_model()
    enc_model1.add_noise(10)
    assert not np.allclose(enc_model1.predict(X), np.mean(y[0]))

    enc_model2 = model2.new_model()
    enc_model2.add_noise(11)
    assert not np.allclose(enc_model2.predict(X), np.mean(y[1]))

    enc_model3 = model3.new_model()
    enc_model3.add_noise(12)
    assert not np.allclose(enc_model3.predict(X), np.mean(y[2]))

    final_model = enc_model1
    final_model.aggregate([enc_model2, enc_model3])
    final_model.remove_noise_models([10, 11, 12])
    assert np.allclose(final_model.predict(X), np.mean(y[0] + y[1] + y[2]))


def test_analytics_variance_aggregation():
    model_data = {"model_type": "analytics", "model_name": "Variance"}

    model1 = analytics_model.SingleModel(model_data)
    model1.fit(X, y[0])

    model2 = analytics_model.SingleModel(model_data)
    model2.fit(X, y[1])

    model3 = analytics_model.SingleModel(model_data)
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), np.var(y[0]))
    assert np.allclose(model2.predict(X), np.var(y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model2])
    assert np.allclose(agg_model.predict(X), np.var(y[0] + y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model3])
    assert np.allclose(agg_model.predict(X), np.var(y[0] + y[2]))

    enc_model1 = model1.new_model()
    enc_model1.add_noise(10)
    assert not np.allclose(enc_model1.predict(X), np.var(y[0]))

    enc_model2 = model2.new_model()
    enc_model2.add_noise(11)
    assert not np.allclose(enc_model2.predict(X), np.var(y[1]))

    enc_model3 = model3.new_model()
    enc_model3.add_noise(12)
    assert not np.allclose(enc_model3.predict(X), np.var(y[2]))

    final_model = enc_model1
    final_model.aggregate([enc_model2, enc_model3])
    final_model.remove_noise_models([10, 11, 12])
    assert np.allclose(final_model.predict(X), np.var(y[0] + y[1] + y[2]))


def test_analytics_std_aggregation():
    model_data = {"model_type": "analytics", "model_name": "Std"}

    model1 = analytics_model.SingleModel(model_data)
    model1.fit(X, y[0])

    model2 = analytics_model.SingleModel(model_data)
    model2.fit(X, y[1])

    model3 = analytics_model.SingleModel(model_data)
    model3.fit(X, y[2])

    assert np.allclose(model1.predict(X), np.std(y[0]))
    assert np.allclose(model2.predict(X), np.std(y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model2])
    assert np.allclose(agg_model.predict(X), np.std(y[0] + y[1]))

    agg_model = model1.new_model()
    agg_model.aggregate([model3])
    assert np.allclose(agg_model.predict(X), np.std(y[0] + y[2]))

    enc_model1 = model1.new_model()
    enc_model1.add_noise(10)
    assert not np.allclose(enc_model1.predict(X), np.std(y[0]))

    enc_model2 = model2.new_model()
    enc_model2.add_noise(11)
    assert not np.allclose(enc_model2.predict(X), np.std(y[1]))

    enc_model3 = model3.new_model()
    enc_model3.add_noise(12)
    assert not np.allclose(enc_model3.predict(X), np.std(y[2]))

    final_model = enc_model1
    final_model.aggregate([enc_model2, enc_model3])
    final_model.remove_noise_models([10, 11, 12])
    assert np.allclose(final_model.predict(X), np.std(y[0] + y[1] + y[2]))
