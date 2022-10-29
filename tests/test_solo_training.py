"""Test training process."""
import json

from feltlabs.algorithm import train
from feltlabs.config import parse_training_args
from feltlabs.core.data import load_data
from feltlabs.core.storage import load_model
from feltlabs.model import load_model

# data = np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 1, 2])
# Right now it uses test dataset

model_def = {
    "model_type": "sklearn",
    "model_name": "LinearRegression",
}


def test_training(tmp_path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"

    input_folder.mkdir()
    output_folder.mkdir()

    # Create custom data file (containing model definition)
    with open(input_folder / "algoCustomData.json", "w") as f:
        json.dump(model_def, f)

    ### Training section ###
    args_str = f"--output_folder {output_folder}"
    args_str += f" --input_folder {input_folder}"
    args_str += f" --solo_training"

    args = parse_training_args(args_str.split())

    # Define extra args with different output folder
    model_name = "model.json"
    model = train.main(args_str.split(), model_name)

    model = load_model(output_folder / model_name)

    # Predict
    data = load_data(args)
    model.predict(data[0])
