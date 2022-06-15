"""Module for loading data provider (Node) config."""
import argparse
from typing import Optional, cast

from feltoken.core.ocean import get_ocean_config


class TrainingConfig:
    model: str
    aggregation_key: bytes
    public_key: bytes
    data_type: str


class AggregationConfig:
    private_key: bytes
    public_key: bytes


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
        "--aggregation_key",
        type=str,
        help="Public key used for encrypting local model for aggregation algorithm.",
    )
    parser.add_argument(
        "--public_key",
        type=str,
        help="Public key used for encrypting rand model for scientis.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="test",
        choices=["test", "csv"],
        help=("Select type of data. For csv last column is used as Y."),
    )
    args = parser.parse_args(args_str)

    conf = get_ocean_config()
    args.public_key = conf["public_key"] if conf["public_key"] else args.public_key
    args.model = conf["model"] if conf["model"] else args.model
    args.data_type = conf["data_type"] if conf["data_type"] else args.model

    args.aggregation_key = bytes.fromhex(args.aggregation_key)
    args.public_key = bytes.fromhex(args.public_key)

    return cast(TrainingConfig, args)


def parse_aggregation_args(args_str: Optional[str] = None) -> AggregationConfig:
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
        "--private_key",
        type=str,
        help="Private key specified for aggregation algorithm used for decrypting models.",
    )
    parser.add_argument(
        "--public_key",
        type=str,
        help="Public key used for encrypting final model for scientis.",
    )
    args = parser.parse_args(args_str)

    conf = get_ocean_config()
    args.public_key = conf["public_key"] if conf["public_key"] else args.public_key

    args.private_key = bytes.fromhex(args.private_key)
    args.public_key = bytes.fromhex(args.public_key)

    return cast(AggregationConfig, args)
