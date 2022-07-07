"""Testing import/export of sklearn models to json."""
import numpy as np
from sklearn.linear_model import LinearRegression

from feltlabs.core.sklearn_to_json import export_model, import_model

X, y = [[0, 0], [1, 1], [2, 2]], [0, 1, 2]


def test_import_export(tmp_path):
    model = LinearRegression()
    model.fit(X, y)

    file_path = tmp_path / "model.json"

    export_model(model, file_path)
    model_bytes = export_model(model)
    with open(file_path, "rb") as f:
        assert f.read() == model_bytes

    im_model = import_model(file_path)

    assert np.array_equal(im_model.coef_, model.coef_)
    assert np.array_equal(im_model.predict(X), model.predict(X))
