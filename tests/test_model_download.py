"""Test model download."""
import json

from nacl.public import PrivateKey

from feltlabs.config import parse_aggregation_args
from feltlabs.core.data import load_models

aggregation_key = PrivateKey.from_seed(b"4" * 32)

# Test model url encrypted by above aggregation key
conf_dict = {
    "model_urls": ["https://github.com/FELT-Labs/feltlabs.py/files/9388706/model.txt"]
}


def test_training(tmp_path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"

    input_folder.mkdir()
    output_folder.mkdir()

    # Create custom data file (containing model URLs)
    with open(input_folder / "algoCustomData.json", "w") as f:
        json.dump(conf_dict, f)

    args_str = f"--output_folder {output_folder}"
    args_str += f" --input_folder {input_folder}"
    args_str += f" --private_key {bytes(aggregation_key).hex()}"
    args_str += f" --download_models"

    # Get aggregation arguments
    args = parse_aggregation_args(args_str.split())
    # Download models from urls
    models = load_models(args)
    assert models[0].predict([[1]]) == 1
