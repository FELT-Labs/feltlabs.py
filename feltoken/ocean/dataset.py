"""Module for publishing demo dataset."""
from ocean_lib.common.agreements.service_types import ServiceTypes
from ocean_lib.services.service import Service
from ocean_lib.web3_internal.currency import to_wei
from ocean_lib.web3_internal.wallet import Wallet

from feltoken.ocean.ocean import get_ocean


def publish_dataset(wallet: Wallet):
    ocean, provider_url = get_ocean()
    data_datatoken = ocean.create_data_token(
        "data1", "data1", wallet, blob=ocean.config.metadata_cache_uri
    )
    data_datatoken.mint(wallet.address, to_wei(100), wallet)
    print(f"data_datatoken.address = '{data_datatoken.address}'")
    # Specify metadata & service attributes for dataset
    data_metadata = {
        "main": {
            "type": "dataset",
            "files": [
                {
                    "url": "https://raw.githubusercontent.com/trentmc/branin/main/branin.arff",
                    "index": 0,
                    "contentType": "text/text",
                }
            ],
            "name": "branin",
            "author": "Trent",
            "license": "CC0",
            "dateCreated": "2019-12-28T10:55:11Z",
        }
    }
    data_service_attributes = {
        "main": {
            "name": "data_dataAssetAccessServiceAgreement",
            "creator": wallet.address,
            "timeout": 3600 * 24,
            "datePublished": "2019-12-28T10:55:11Z",
            "cost": 1.0,  # <don't change, this is obsolete>
        }
    }

    # Calc data service compute descriptor
    data_compute_service = Service(
        service_endpoint=provider_url,
        service_type=ServiceTypes.CLOUD_COMPUTE,
        attributes=data_service_attributes,
    )

    # Publish metadata and service info on-chain
    data_ddo = ocean.assets.create(
        metadata=data_metadata,
        publisher_wallet=wallet,
        services=[data_compute_service],
        data_token_address=data_datatoken.address,
    )
    print(f"data did = '{data_ddo.did}'")
    return data_ddo
