"""Module for aggregating outputs of local training."""
from typing import List, Optional

from feltlabs.config import AggregationConfig, parse_aggregation_args
from feltlabs.core.data import load_models
from feltlabs.core.ocean import save_output
from feltlabs.core.storage import encrypt_model


def main(
    args_str: Optional[List[str]] = None,
    config: Optional[AggregationConfig] = None,
    output_name: str = "model",
):
    """Main function for executing from command line.

    Args:
        args_str: list with string arguments or None if using command line
        config: config object is used instead of arg parser if specified
        output_name: name of output model
    """
    if config is None:
        config = parse_aggregation_args(args_str)
    # Load models
    models = load_models(config)
    if len(models) < config.min_models:
        raise Exception(
            f"Not enough models for aggregation, loaded {len(models)} models (required {config.min_models})."
        )
    # Aggregate
    model, *other_models = models
    model.aggregate(other_models)
    # Encrypt final model using scientist public key if provided
    if config.public_key:
        model = encrypt_model(model, config.public_key)
    else:
        model = model.export_model()
    # Save model (bytes) into output
    save_output(output_name, model, config)
    print("Aggregation finished.")
    return model


if __name__ == "__main__":
    main()
