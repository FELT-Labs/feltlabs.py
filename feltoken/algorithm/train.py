"""Module for executing model training."""
from typing import Optional

from feltoken.config import parse_training_args
from feltoken.core.aggregation import random_model, sum_models
from feltoken.core.data import load_data
from feltoken.core.ocean import save_output
from feltoken.core.storage import encrypt_model, load_model
from feltoken.core.training import train_model


def main(args_str: Optional[list[str]] = None, output_name: str = "model"):
    """Main function for exectuting from command line.

    Args:
        args_str: list with string arguments or None if using command line
        output_name: name of output model
    """
    args = parse_training_args(args_str)
    # Load data and model
    model = load_model(args.input_folder / args.custom_data)
    data = load_data(args)
    # Train model
    model = train_model(model, data)
    # Add randomness to model and encrypt using public key for aggregation
    rand_model = random_model(model)
    model = sum_models([model, rand_model])

    enc_model = encrypt_model(model, args.aggregation_key)
    enc_rand_model = encrypt_model(rand_model, args.public_key)

    # Save models into output
    save_output(output_name, enc_model, args)
    save_output(f"{output_name}_rand", enc_rand_model, args)
    return enc_model, enc_rand_model


if __name__ == "__main__":
    main()
