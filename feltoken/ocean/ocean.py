from ocean_lib.common.agreements.service_types import ServiceTypes
from ocean_lib.data_provider.data_service_provider import DataServiceProvider
from ocean_lib.example_config import ExampleConfig
from ocean_lib.models.compute_input import ComputeInput
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.services.service import Service
from ocean_lib.web3_internal.constants import ZERO_ADDRESS
from ocean_lib.web3_internal.currency import to_wei

from feltoken.ocean.algorithm import alg_metadata, get_attributes


def get_ocean():
    config = ExampleConfig.get_config()
    ocean = Ocean(config)
    provider_url = DataServiceProvider.get_url(ocean.config)
    return (ocean, provider_url)


def publish_algorithm(wallet):
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
        metadata=alg_metadata,
        publisher_wallet=wallet,
        services=[alg_access_service],
        data_token_address=alg_datatoken.address,
    )
    print(f"alg did = '{alg_ddo.did}'")
    return alg_ddo, alg_datatoken


def start_job(data_did, alg_did, alg_datatoken, wallet, model_id):
    ocean, provider_url = get_ocean()
    # make sure we operate on the updated and indexed metadata_cache_uri versions
    data_ddo = ocean.assets.resolve(data_did)
    alg_ddo = ocean.assets.resolve(alg_did)

    compute_service = data_ddo.get_service("compute")
    algo_service = alg_ddo.get_service("access")
    # order & pay for dataset
    dataset_order_requirements = ocean.assets.order(
        data_did, wallet.address, service_type=compute_service.type
    )
    data_order_tx_id = ocean.assets.pay_for_service(
        ocean.web3,
        dataset_order_requirements.amount,
        dataset_order_requirements.data_token_address,
        data_did,
        compute_service.index,
        ZERO_ADDRESS,
        wallet,
        dataset_order_requirements.computeAddress,
    )

    # order & pay for algo
    algo_order_requirements = ocean.assets.order(
        alg_did, wallet.address, service_type=algo_service.type
    )
    alg_order_tx_id = ocean.assets.pay_for_service(
        ocean.web3,
        algo_order_requirements.amount,
        algo_order_requirements.data_token_address,
        alg_did,
        algo_service.index,
        ZERO_ADDRESS,
        wallet,
        algo_order_requirements.computeAddress,
    )

    compute_inputs = [ComputeInput(data_did, data_order_tx_id, compute_service.index)]
    job_id = ocean.compute.start(
        compute_inputs,
        wallet,
        algorithm_did=alg_did,
        algorithm_tx_id=alg_order_tx_id,
        algorithm_data_token=alg_datatoken.address,
        algouserdata={"_id": model_id},
    )
    print(f"Started compute job with id: {job_id}")
    return job_id
