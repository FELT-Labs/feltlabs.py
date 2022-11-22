"""Test training process."""
from pathlib import Path

import pytest
from nacl.public import PrivateKey

from feltlabs.algorithm import aggregate, train
from feltlabs.config import parse_training_args
from feltlabs.core.data import load_data
from feltlabs.core.json_handler import json_dump
from feltlabs.core.storage import load_model

# data = np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 1, 2])
# Right now it uses test dataset

aggregation_key = PrivateKey.generate()
scientist_key = PrivateKey.generate()

model_def = {
    "model_type": "sklearn",
    "model_name": "LinearRegression",
}


def test_training(tmp_path: Path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output" / "fake_did"
    output_folder2 = tmp_path / "output2"

    input_folder.mkdir()
    output_folder.mkdir(parents=True)
    output_folder2.mkdir()

    # Create custom data file (containing model definition)
    with open(input_folder / "algoCustomData.json", "wb") as f:
        f.write(json_dump(model_def))

    enc_models, seeds = [], []

    ### Training section ###
    args_str = f"--output_folder {output_folder}"
    args_str += f" --input_folder {input_folder}"
    args_str += f" --aggregation_key {bytes(aggregation_key.public_key).hex()}"

    # Define extra args with different output folder
    args = parse_training_args(args_str.split())

    for i in range(2):
        args_str_final = f"{args_str} --seed {i}"
        enc_model = train.main(args_str_final.split(), output_name=f"{i}")

        enc_models.append(enc_model)
        seeds.append(i)

    ### Aggregation section ###
    args_str = f"--output_folder {output_folder2}"
    args_str += f" --input_folder {output_folder.parent}"
    args_str += f" --private_key {bytes(aggregation_key).hex()}"

    enc_final_model = aggregate.main(args_str.split(), output_name="final_model")

    ### Test final results ###
    final_model = load_model(enc_final_model)
    final_model.remove_noise_models(seeds)

    # Predict
    data = load_data(args)
    final_model.predict(data[0])

    ### Aggregation section ###
    args_str = f"--output_folder {output_folder2}"
    args_str += f" --input_folder {output_folder}"
    args_str += f" --private_key {bytes(aggregation_key).hex()}"
    args_str += f" --min_models 3"

    with pytest.raises(Exception):
        # Should fail because at least 3 models are required
        enc_final_model = aggregate.main(args_str.split(), output_name="final_model")
