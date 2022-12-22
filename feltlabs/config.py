"""Module for loading data provider (Node) config."""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

OUTPUT_FOLDER = Path("/data/outputs")
INPUT_FOLDER = Path("/data/inputs/")
CUSTOM_DATA = "algoCustomData.json"


@dataclass
class OceanConfig:
    input_folder: Path = INPUT_FOLDER
    output_folder: Path = OUTPUT_FOLDER
    custom_data_path: Path = INPUT_FOLDER / CUSTOM_DATA
    custom_data: str = CUSTOM_DATA


@dataclass
class TrainingConfig(OceanConfig):
    aggregation_key: bytes = bytes(0)
    data_type: str = "test"
    seed: int = 42
    target_column: int = -1
    solo_training: bool = False
    experimental: bool = False


@dataclass
class AggregationConfig(OceanConfig):
    private_key: bytes = bytes(0)
    public_key: Optional[bytes] = None
    download_models: bool = False
    min_models: int = 2


def _help_exit(parser, error_msg=None):
    """Print help of parser and quit the script."""
    parser.print_help()
    if error_msg:
        print(f"\nERROR: {error_msg}")
    sys.exit(2)


def _add_ocean_config(config: OceanConfig) -> Dict[str, str]:
    """Load json file containing algorithm's custom data and add them to config.

    Args:
        config: ocean config containing output path

    Returns:
        dict representing loaded JSON file
    """
    config.custom_data_path = config.input_folder / config.custom_data
    if not config.custom_data_path.exists():
        return {}

    with config.custom_data_path.open("r") as f:
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


def parse_training_args(args_str: Optional[List[str]] = None) -> TrainingConfig:
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str: list with string arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="""
            Script for training models, possible to execute from command line.
            At least one of the flags --solo_training or --aggregation_key KEY must be set.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = _ocean_parser(parser)
    # Fixed algorithm params (set as part of deployed algorithm)
    parser.add_argument(
        "--aggregation_key",
        type=str,
        help="Public key used for encrypting local model for aggregation algorithm.",
    )
    parser.add_argument(
        "--solo_training",
        action="store_true",
        default=False,
        help="If true (flag included), it will run training on single dataset without encryption.",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        default=False,
        help="If true, allow usage of experimental models (which might be less secure).",
    )
    # Changable arguments (can be changed via algorithm custom data)
    parser.add_argument(
        "--data_type",
        type=str,
        default="test",
        choices=["test", "csv", "pickle"],
        help="Select type of data. For csv last column is used as Y.",
    )
    parser.add_argument(
        "--target_column",
        type=int,
        default=-1,
        help="Select index of target column.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for randomness generation. It should be different for every run.",
    )
    args = parser.parse_args(args_str)

    conf = _add_ocean_config(cast(OceanConfig, args))
    args.data_type = conf["data_type"] if "data_type" in conf else args.data_type
    args.seed = conf["seed"] if "seed" in conf else args.seed
    args.target_column = (
        conf["target_column"] if "target_column" in conf else args.target_column
    )

    if not args.solo_training and not args.aggregation_key:
        # At least one of solo_training or aggregation_key must be set (else exit)
        _help_exit(
            parser, "At least one of --solo_training, --aggregation_key must be set"
        )

    if not args.solo_training:
        args.aggregation_key = bytes.fromhex(args.aggregation_key)

    return cast(TrainingConfig, args)


def parse_aggregation_args(args_str: Optional[List[str]] = None) -> AggregationConfig:
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str: list with string arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Script for training models, possible to execute from command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = _ocean_parser(parser)
    # Fixed algorithm params (set as part of deployed algorithm)
    parser.add_argument(
        "--private_key",
        type=str,
        help="Private key specified for aggregation algorithm used for decrypting models.",
    )
    parser.add_argument(
        "--min_models",
        type=int,
        default=2,
        help="Minimum number of models required for aggregation (agg fails otherwise).",
    )
    parser.add_argument(
        "--download_models",
        help="If true (flag included), models will be downloaded from provided URLs",
        action="store_true",
        default=False,
    )
    # Changable arguments (can be changed via algorithm custom data)
    parser.add_argument(
        "--public_key",
        type=str,
        help="Public key used for encrypting final model for scientist.",
        default=None,
    )
    args = parser.parse_args(args_str)

    conf = _add_ocean_config(cast(OceanConfig, args))
    args.public_key = conf["public_key"] if "public_key" in conf else args.public_key

    if not args.private_key:
        # Private key is required
        _help_exit(parser, "Private key is not defined.")

    args.private_key = bytes.fromhex(args.private_key)
    if args.public_key:
        args.public_key = bytes.fromhex(args.public_key)

    return cast(AggregationConfig, args)
