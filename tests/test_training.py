"""Test training process."""
import json

from nacl.public import PrivateKey

from feltlabs.algorithm import aggregate, train
from feltlabs.config import parse_training_args
from feltlabs.core.aggregation import random_model, remove_noise_models
from feltlabs.core.cryptography import decrypt_nacl
from feltlabs.core.data import load_data
from feltlabs.core.ocean import save_output
from feltlabs.core.storage import load_model

# data = np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 1, 2])

aggregation_key = PrivateKey.generate()
scientist_key = PrivateKey.generate()

model_def = {"model_name": "LinearRegression"}


def test_training(tmp_path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    output_folder2 = tmp_path / "output2"

    input_folder.mkdir()
    output_folder.mkdir()
    output_folder2.mkdir()

    # Create custom data file (containing model definition)
    with open(input_folder / "algoCustomData.json", "w") as f:
        json.dump(model_def, f)

    enc_models, seeds = [], []

    ### Training section ###
    args_str = f"--output_folder {output_folder}"
    args_str += f" --input_folder {input_folder}"
    args_str += f" --aggregation_key {bytes(aggregation_key.public_key).hex()}"

    # Define extra args with different output folder
    args = parse_training_args(args_str.split())
    args.output_folder = output_folder2

    for i in range(2):
        args_str_final = f"{args_str} --seed {i}"
        enc_model = train.main(args_str_final.split(), f"model_{i}")

        save_output(f"model_{i}", enc_model, args)

        enc_models.append(enc_model)
        seeds.append(i)

    ### Aggregation section ###
    args_str = f"--output_folder {output_folder2}"
    args_str += f" --input_folder {output_folder2}"
    args_str += f" --private_key {bytes(aggregation_key).hex()}"

    enc_final_model = aggregate.main(args_str.split(), "final_model")

    ### Test final results ###
    final_model = load_model(enc_final_model)
    rand_models = [random_model(final_model, s) for s in seeds]
    model = remove_noise_models(final_model, rand_models)

    # Predict
    data = load_data(args)
    model.predict(data[0])
