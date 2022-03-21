"""Module for executing model training."""
import argparse

from feltoken.core.data import load_data
from feltoken.core.storage import export_model, load_model


def train_model(model, data):
    model.fit(data[0], data[1])


def parse_args(args_str=None):
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str (str): string with arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Script for training models, possible to execute from command line."
    )
    parser.add_argument(
        "--model",
        type=int,
        help="Path to model file which should be loaded and trained.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help="Path to CSV file with data. Last column is considered as Y.",
    )
    parser.add_argument(
        "--output_path",
        type=int,
        help="Path to store the final model file.",
    )
    args = parser.parse_args(args_str)

    return args


def main():
    """Main function for exectuting from command line."""
    args = parse_args()
    model = load_model(args.model)

    data = load_data(args.data)

    model = train_model(model, data)
    export_model(model, args.output_path)


if __name__ == "__main__":
    main()
