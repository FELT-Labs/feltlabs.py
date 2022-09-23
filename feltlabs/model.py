"""Module for handling final models - loading model, exporting model to pickle format."""
import argparse
import pickle
from pathlib import Path

from feltlabs.core.storage import load_model


def main():
    """Load model from JSON file and save it as pickle file based on arguments."""

    parser = argparse.ArgumentParser(
        description="Script for training models, possible to execute from command line."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input model file in JSON format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output file.",
    )
    args = parser.parse_args()

    model = load_model(args.input)
    pickle.dump(model, args.output)


if __name__ == "__main__":
    main()
