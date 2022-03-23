"""Module for publishing algorithm.

This is mainly for demonstration purposes, so we know how to use ocean
and for testing the compute to data.

It publishes algorithm using PRIVATE_KEY variable from .env
You also have to update FILE_PROVIDER_URL variable in .env

Run:
    python algorithm.py
"""
import os
import sys

from dotenv import load_dotenv
from ocean_lib.common.agreements.service_types import ServiceTypes
from ocean_lib.services.service import Service
from ocean_lib.web3_internal.currency import to_wei
from ocean_lib.web3_internal.wallet import Wallet

from feltoken.ocean.ocean import get_ocean, get_wallet


def get_metadata(file_provider_url: str):
    return {
        "main": {
            "type": "algorithm",
            "algorithm": {
                "language": "python",
                "format": "docker-image",
                "version": "0.1",
                "container": {
                    "entrypoint": "python -m pip install git+https://github.com/FELToken/feltoken.py/tree/ocean-integration; feltoken-train --model $ALGO --data test --output_model /data/outputs/result",
                    "image": "oceanprotocol/algo_dockers",
                    "tag": "python-branin",
                },
            },
            "files": [
                {
                    "url": f"{file_provider_url}/model",
                    "index": 0,
                    "contentType": "text/text",
                }
            ],
            "name": "gpr",
            "author": "Trent",
            "license": "CC0",
            "dateCreated": "2020-01-28T10:55:11Z",
        }
    }


def get_attributes(address: str):
    return {
        "main": {
            "name": "ALG_dataAssetAccessServiceAgreement",
            "creator": address,
            "timeout": 3600 * 24,
            "datePublished": "2020-01-28T10:55:11Z",
            "cost": 1.0,  # <don't change, this is obsolete>
        }
    }


def publish_algorithm(file_provider_url: str, wallet: Wallet):
    ocean, provider_url = get_ocean()
    # Publish alg datatoken
    alg_datatoken = ocean.create_data_token(
        "alg1", "alg1", wallet, blob=ocean.config.metadata_cache_uri
    )
    alg_datatoken.mint(wallet.address, to_wei(100), wallet)
    print(f"alg_datatoken.address = '{alg_datatoken.address}'")

    # Calc alg service access descriptor. We use the same service provider as data
    alg_access_service = Service(
        service_endpoint=provider_url,
        service_type=ServiceTypes.CLOUD_COMPUTE,
        attributes=get_attributes(wallet.address),
    )

    # Publish metadata and service info on-chain
    alg_ddo = ocean.assets.create(
        metadata=get_metadata(file_provider_url),
        publisher_wallet=wallet,
        services=[alg_access_service],
        data_token_address=alg_datatoken.address,
    )
    print(f"alg did = '{alg_ddo.did}'")
    return alg_ddo, alg_datatoken


if __name__ == "__main__":
    load_dotenv()
    wallet = get_wallet(os.getenv("PRIVATE_KEY"))
    publish_algorithm(os.getenv("FILE_PROVIDER_URL", ""), wallet)
