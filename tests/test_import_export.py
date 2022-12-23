"""Testing import/export to json."""
from pathlib import Path
from typing import cast

import numpy as np

from feltlabs.core.models import analytics_model, sklearn_model
from feltlabs.core.storage import load_model

X, y = [[0, 0], [1, 1], [2, 2]], [0, 1, 2]


def test_sklearn_linreg_import_export(tmp_path: Path):
    model_data = {"model_type": "sklearn", "model_name": "LinearRegression"}

    model = sklearn_model.Model(model_data)
    model.fit(X, y)

    file_path = tmp_path / "model.json"

    model.export_model(file_path)
    model_bytes = model.export_model()

    with open(file_path, "rb") as f:
        assert f.read() == model_bytes

    im_model = cast(sklearn_model.Model, load_model(file_path))

    assert np.array_equal(im_model.model.coef_, model.model.coef_)
    assert np.array_equal(im_model.predict(X), model.predict(X))


def test_sklearn_ridge_import_export(tmp_path: Path):
    model_data = {
        "model_type": "sklearn",
        "model_name": "Ridge",
        "init_params": {"alpha": 0.1},
    }

    model = sklearn_model.Model(model_data)
    model.fit(X, y)

    file_path = tmp_path / "model.json"

    model.export_model(file_path)
    model_bytes = model.export_model()

    with open(file_path, "rb") as f:
        assert f.read() == model_bytes

    im_model = cast(sklearn_model.Model, load_model(file_path))

    assert np.array_equal(im_model.model.coef_, model.model.coef_)
    assert np.array_equal(im_model.predict(X), model.predict(X))


def test_analytics_import_export(tmp_path: Path):
    model_data = {"model_type": "analytics", "model_name": "Sum"}

    model = analytics_model.Model(model_data)
    model.fit(X, y)

    file_path = tmp_path / "model.json"

    model.export_model(file_path)
    model_bytes = model.export_model()

    with open(file_path, "rb") as f:
        assert f.read() == model_bytes

    im_model = load_model(file_path)

    im_model = cast(analytics_model.Model, im_model)
    assert im_model.models[0].__dict__ == model.models[0].__dict__
    assert np.array_equal(im_model.predict(X), model.predict(X))


def test_multi_analytics_import_export(tmp_path: Path):
    model_data = [
        {"model_type": "analytics", "model_name": "Sum"},
        {"model_type": "analytics", "model_name": "Mean"},
    ]

    model = analytics_model.Model(model_data)
    model.fit(X, y)

    file_path = tmp_path / "model.json"

    model.export_model(file_path)
    model_bytes = model.export_model()

    with open(file_path, "rb") as f:
        assert f.read() == model_bytes

    im_model = load_model(file_path)

    im_model = cast(analytics_model.Model, im_model)
    assert im_model.models[0].__dict__ == model.models[0].__dict__
    assert np.array_equal(im_model.predict(X), model.predict(X))
