"""Testing import/export of sklearn models to json."""
import numpy as np

from feltlabs.core.models import sklearn_model
from feltlabs.core.storage import load_model

X, y = [[0, 0], [1, 1], [2, 2]], [0, 1, 2]


def test_import_export(tmp_path):
    model_data = {"model_type": "sklearn", "model_name": "LinearRegression"}

    model = sklearn_model.Model(model_data)
    model.fit(X, y)

    file_path = tmp_path / "model.json"

    model.export_model(file_path)
    model_bytes = model.export_model()

    with open(file_path, "rb") as f:
        assert f.read() == model_bytes

    im_model = load_model(file_path)

    assert np.array_equal(im_model.model.coef_, model.model.coef_)
    assert np.array_equal(im_model.predict(X), model.predict(X))
