"""Module for loading data provider (Node) config."""
import argparse
from typing import Optional, cast


class TrainingConfig:
    model: str
    public_key: bytes
    data_type: str


def parse_training_args(args_str: Optional[str] = None) -> TrainingConfig:
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str: string with arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Script for training models, possible to execute from command line."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model config to use.",
    )
    parser.add_argument(
        "--public_key",
        type=str,
        help="Public key used for encrypting local model for aggregation algorithm.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="test",
        choices=["test", "csv"],
        help=("Select type of data. For csv last column is used as Y."),
    )
    args = parser.parse_args(args_str)
    args.public_key = bytes.fromhex(args.public_key)

    return cast(TrainingConfig, args)
