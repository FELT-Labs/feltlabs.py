"""Module for executing model training."""
from typing import Optional

from feltlabs.config import parse_training_args
from feltlabs.core.data import load_data
from feltlabs.core.ocean import save_output
from feltlabs.core.storage import encrypt_model, load_model


def main(args_str: Optional[list[str]] = None, output_name: str = "model"):
    """Main function for exectuting from command line.

    Args:
        args_str: list with string arguments or None if using command line
        output_name: name of output model
    """
    args = parse_training_args(args_str)
    # Load model
    model = load_model(args.custom_data_path)
    # Load data and train model
    X, y = load_data(args)
    model.fit(X, y)

    if args.solo_training:
        model = model.export_model()
    else:
        # Add randomness to model
        model.add_noise(args.seed)
        # Encrypt model using public key of aggregation
        model = encrypt_model(model, args.aggregation_key)

    # Save models into output
    save_output(output_name, model, args)
    print("Training finished.")
    return model


if __name__ == "__main__":
    main()
