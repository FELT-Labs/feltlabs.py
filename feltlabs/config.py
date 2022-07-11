"""Module for loading data provider (Node) config."""
import argparse
import json
from pathlib import Path
from typing import Optional, cast

OUTPUT_FOLDER = Path("/data/outputs")
INPUT_FOLDER = Path("/data/inputs/")
CUSTOM_DATA = "algoCustomData.json"


class OceanConfig:
    input_folder: Path
    output_folder: Path
    custom_data: str


class TrainingConfig(OceanConfig):
    aggregation_key: bytes
    data_type: str
    seed: int


class AggregationConfig(OceanConfig):
    private_key: bytes
    public_key: bytes


def _get_ocean_config(config: OceanConfig) -> dict[str, str]:
    """Load json file containing algorithm's custom data.

    Args:
        config: ocean config containing output path

    Returns:
        dict representing loaded JSON file
    """
    file = config.input_folder / config.custom_data
    if not file.exists():
        return {}

    with file.open("r") as f:
        return json.load(f)


def _ocean_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for parsing ocean path configs."""
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=OUTPUT_FOLDER,
        help="Folder for storing outputs.",
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        default=INPUT_FOLDER,
        help="Folder containing input data.",
    )
    parser.add_argument(
        "--custom_data",
        type=str,
        default=CUSTOM_DATA,
        help="Name of custom data file",
    )
    return parser


def parse_training_args(args_str: Optional[list[str]] = None) -> TrainingConfig:
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str: list with string arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Script for training models, possible to execute from command line."
    )
    parser = _ocean_parser(parser)
    parser.add_argument(
        "--aggregation_key",
        type=str,
        help="Public key used for encrypting local model for aggregation algorithm.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="test",
        choices=["test", "csv"],
        help="Select type of data. For csv last column is used as Y.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for randomness generation. It should be different for every run.",
    )
    args = parser.parse_args(args_str)

    conf = _get_ocean_config(cast(OceanConfig, args))
    args.data_type = conf["data_type"] if "data_type" in conf else args.data_type
    args.seed = conf["seed"] if "seed" in conf else args.seed
    args.aggregation_key = bytes.fromhex(args.aggregation_key)

    return cast(TrainingConfig, args)


def parse_aggregation_args(args_str: Optional[list[str]] = None) -> AggregationConfig:
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str: list with string arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Script for training models, possible to execute from command line."
    )
    parser = _ocean_parser(parser)
    parser.add_argument(
        "--private_key",
        type=str,
        help="Private key specified for aggregation algorithm used for decrypting models.",
    )
    parser.add_argument(
        "--public_key",
        type=str,
        help="Public key used for encrypting final model for scientis.",
        default=None,
    )
    args = parser.parse_args(args_str)

    conf = _get_ocean_config(cast(OceanConfig, args))
    args.public_key = conf["public_key"] if "public_key" in conf else args.public_key

    args.private_key = bytes.fromhex(args.private_key)
    if args.public_key:
        args.public_key = bytes.fromhex(args.public_key)

    return cast(AggregationConfig, args)
