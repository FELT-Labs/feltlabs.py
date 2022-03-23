"""Module for approving algorithm for given dataset.

This is mainly for demonstration purposes, so we know how to use ocean
and for testing the compute to data.

It allows given algorithm for running on data
(data should be published using PRIVATE_KEY)

Run:
    python allow_algorithm.py data_did algorithm_did

Example:
    python feltoken/ocean/allow_algorithm.py did:op:05EA0f00CA20053B1368f9DfF010e30854504a7C did:op:6acB4766F05c7c8E2bea1959Ed8322800546C407
"""
import os
import sys

from dotenv import load_dotenv
from ocean_lib.assets.asset_resolver import resolve_asset
from ocean_lib.assets.trusted_algorithms import add_publisher_trusted_algorithm
from ocean_lib.web3_internal.wallet import Wallet

from feltoken.ocean.ocean import get_ocean, get_wallet


def allow_algorithm(data_did: str, alg_did: str, wallet: Wallet):
    ocean, _ = get_ocean()
    data_ddo = resolve_asset(
        data_did, metadata_cache_uri=ocean.config.metadata_cache_uri
    )
    add_publisher_trusted_algorithm(data_ddo, alg_did, ocean.config.metadata_cache_uri)
    ocean.assets.update(data_ddo, publisher_wallet=wallet)


if __name__ == "__main__":
    load_dotenv()
    wallet = get_wallet(os.getenv("PRIVATE_KEY"))
    allow_algorithm(sys.argv[1], sys.argv[2], wallet)
