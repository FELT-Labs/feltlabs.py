"""Testing loading of data."""
import numpy as np
from sklearn.linear_model import LinearRegression

from feltlabs.config import parse_training_args
from feltlabs.core.data import load_data

# Test data both with and without header
csv_data = [
    "column_1,col_2,c3\n1,2,3\n0,2,3\n1,1,4\n0,2,5\n1,1,6",
    "1,2,3\n0,2,3\n1,1,4\n0,2,5\n1,1,6",
]


def test_csv_load(tmp_path):
    # Create CSV files
    for i in range(2):
        file_path = tmp_path / f"{i}"
        with open(file_path, "w") as f:
            f.write(csv_data[i])

    args_str = f" --data_type csv --input_folder {tmp_path} --aggregation_key fe"
    # Define extra args with different output folder
    args = parse_training_args(args_str.split())
    X, y = load_data(args)
    print(y)

    assert len(X) == len(y), "Size of X and y are not matching"
    assert len(X) == 2 * 5, "Not all data loaded"
    assert np.all(X[:5] == np.array([[1, 2], [0, 2], [1, 1], [0, 2], [1, 1]]))
    assert np.all(y[:5] == np.array([3, 3, 4, 5, 6]))

    args_str = f" --data_type csv --target_column 0 --input_folder {tmp_path} --aggregation_key fe"
    # Define extra args with different output folder
    args = parse_training_args(args_str.split())
    X, y = load_data(args)
    assert len(X) == len(y), "Size of X and y are not matching"
    assert len(X) == 2 * 5, "Not all data loaded"
    assert np.all(X[:5] == np.array([[2, 3], [2, 3], [1, 4], [2, 5], [1, 6]]))
    assert np.all(y[:5] == np.array([1, 0, 1, 0, 1]))

    # Try if training works
    model = LinearRegression()
    model.fit(X, y)
