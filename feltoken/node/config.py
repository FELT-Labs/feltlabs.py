"""Module for loading data provider (Node) config."""
import argparse
import os
from getpass import getpass
from typing import Optional, cast

from dotenv import load_dotenv

# Load dotenv at the beginning of the program
load_dotenv()


KEYS = {
    "main": os.getenv("PRIVATE_KEY"),
    "node1": os.getenv("NODE1_PRIVATE_KEY"),
    "node2": os.getenv("NODE2_PRIVATE_KEY"),
}


class Config:
    """Config class type."""

    chain: int
    contract: str
    account: str
    data: str
    output_model: str


def _verify_config(config: Config) -> Config:
    """Verify if provided values are correct."""
    assert config.chain in [
        1337,
        80001,
        137,
    ], "Invalid chain id or chain id is not supported (suppoerted: 1337, 137, 80001)"
    assert len(config.contract) == 42, "The contract address has invalid length."

    if config.account in KEYS and KEYS[config.account] is not None:
        config.account = KEYS[config.account]
    else:
        config.account = getpass(
            "Please provide your private key (exported from MetaMask):"
        )

    # Get web3 token if missing
    if "WEB3_STORAGE_TOKEN" not in os.environ or not os.getenv("WEB3_STORAGE_TOKEN"):
        os.environ["WEB3_STORAGE_TOKEN"] = getpass(
            "Please input your web3.storage API token:"
        )

    return config


def parse_args(args_str: Optional[str] = None) -> Config:
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str: string with arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Data provider worker script managing the trainig."
    )
    parser.add_argument(
        "--chain",
        type=int,
        help="Chain Id of chain to which should be the worker connected.",
    )
    parser.add_argument("--contract", type=str, help="Contract address")
    parser.add_argument(
        "--account",
        type=str,
        default="main",
        help="Name of account to use as specified in .env (main, node1, node2)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help=("Path to CSV file with data. Last column is considered as Y."),
    )
    args = parser.parse_args(args_str)

    return _verify_config(cast(Config, args))
