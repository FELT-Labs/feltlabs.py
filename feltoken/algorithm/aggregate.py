"""Module for aggregating outputs of local training."""
from typing import Optional

from feltoken.config import parse_aggregation_args
from feltoken.core.aggregation import aggregate_models
from feltoken.core.data import load_models
from feltoken.core.ocean import save_output
from feltoken.core.storage import encrypt_model


def main(args_str: Optional[list[str]] = None, output_name: str = "model"):
    """Main function for exectuting from command line.

    Args:
        args_str: list with string arguments or None if using command line
        output_name: name of output model
    """
    args = parse_aggregation_args(args_str)
    # Load models
    models = load_models(args)
    # Aggregate
    model = aggregate_models(models)
    # Add randomness to model and encrypt using public key for aggregation
    enc_model = encrypt_model(model, args.public_key)
    # Save model into output
    save_output(output_name, enc_model, args)
    return enc_model


if __name__ == "__main__":
    main()
