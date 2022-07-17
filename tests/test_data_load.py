"""Testing loading of data."""
from sklearn.linear_model import LinearRegression

from feltlabs.config import parse_training_args
from feltlabs.core.data import load_data

csv_data = "1,2,3\n0,2,3\n1,1,4\n0,2,5\n1,1,6"


def test_csv_load(tmp_path):
    # Create CSV files
    for i in range(2):
        file_path = tmp_path / f"{i}"
        with open(file_path, "w") as f:
            f.write(csv_data)

    args_str = f" --data_type csv --input_folder {tmp_path} --aggregation_key fe"
    # Define extra args with different output folder
    args = parse_training_args(args_str.split())
    X, y = load_data(args)

    assert len(X) == len(y), "Size of X and y are not matching"
    assert len(X) == 2 * 5, "Not all data loaded"

    # Try if training works
    model = LinearRegression()
    model.fit(X, y)
