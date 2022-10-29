"""Module for aggregating outputs of local training."""
from typing import Optional

from feltlabs.config import parse_aggregation_args
from feltlabs.core.data import load_models
from feltlabs.core.ocean import save_output
from feltlabs.core.storage import encrypt_model


def main(args_str: Optional[list[str]] = None, output_name: str = "model"):
    """Main function for exectuting from command line.

    Args:
        args_str: list with string arguments or None if using command line
        output_name: name of output model
    """
    args = parse_aggregation_args(args_str)
    # Load models
    models = load_models(args)
    if len(models) < args.min_models:
        raise Exception(
            f"Not enough models for aggregation, loaded {len(models)} models (required {args.min_models})."
        )
    # Aggregate
    model, *other_models = models
    model.aggregate(other_models)
    # Encrypt final model using scientist public key if provided
    if args.public_key:
        model = encrypt_model(model, args.public_key)
    else:
        model = model.export_model()
    # Save model (bytes) into output
    save_output(output_name, model, args)
    print("Aggregation finieshed.")
    return model


if __name__ == "__main__":
    main()
