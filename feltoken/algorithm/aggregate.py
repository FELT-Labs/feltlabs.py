"""Module for aggregating outputs of local training."""
from feltoken.algorithm.config import parse_aggregation_args
from feltoken.core.aggregation import sum_models
from feltoken.core.data import load_models
from feltoken.core.ocean import save_output
from feltoken.core.storage import encrypt_model


def main():
    """Main function for exectuting from command line."""
    args = parse_aggregation_args()
    # Load models
    models = load_models(args)
    # Aggregate
    model = sum_models(models)
    # Add randomness to model and encrypt using public key for aggregation
    enc_model = encrypt_model(model, args.public_key)
    # Save model into output
    save_output("model", enc_model)


if __name__ == "__main__":
    main()
