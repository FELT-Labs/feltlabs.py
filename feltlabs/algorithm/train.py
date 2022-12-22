"""Module for executing model training."""
from typing import List, Optional

from feltlabs.config import TrainingConfig, parse_training_args
from feltlabs.core.data import load_data
from feltlabs.core.ocean import save_output
from feltlabs.core.storage import encrypt_model, load_model


def main(
    args_str: Optional[List[str]] = None,
    config: Optional[TrainingConfig] = None,
    output_name: str = "model",
):
    """Main function for executing from command line.

    Args:
        args_str: list with string arguments or None if using command line
        config: config object is used instead of arg parser if specified
        output_name: name of output model
    """
    if config is None:
        config = parse_training_args(args_str)
    # Load model
    model = load_model(config.custom_data_path, config.experimental)
    # Load data and train model
    X, y = load_data(config)
    model.fit(X, y)

    if config.solo_training:
        model = model.export_model()
    else:
        # Add randomness to model
        model.add_noise(config.seed)
        # Encrypt model using public key of aggregation
        model = encrypt_model(model, config.aggregation_key)

    # Save models into output
    save_output(output_name, model, config)
    print("Training finished.")
    return model


if __name__ == "__main__":
    main()
