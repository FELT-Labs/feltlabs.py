from pathlib import Path

from ocean_lib.assets.asset_resolver import resolve_asset
from ocean_lib.data_provider.data_service_provider import DataServiceProvider
from ocean_lib.example_config import ExampleConfig
from ocean_lib.models.compute_input import ComputeInput
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.web3_internal.constants import ZERO_ADDRESS
from ocean_lib.web3_internal.wallet import Wallet

LOGS = Path(__file__).parent
LOGS.mkdir(exist_ok=True)

OCEAN = None


def get_ocean():
    global OCEAN
    if not OCEAN:
        config = ExampleConfig.get_config()
        OCEAN = Ocean(config)
    provider_url = DataServiceProvider.get_url(OCEAN.config)
    return (OCEAN, provider_url)


def get_wallet(private_key):
    ocean, _ = get_ocean()
    return Wallet(
        ocean.web3,
        private_key,
        ocean.config.block_confirmations,
        ocean.config.transaction_timeout,
    )


def resolve_did(did):
    ocean, _ = get_ocean()
    return resolve_asset(did, metadata_cache_uri=ocean.config.metadata_cache_uri)


def get_algorithm(algorithm_did):
    """Get already deployed algorithm or publish new."""
    ocean, _ = get_ocean()
    # Load algorithm DDO or publish alg
    alg_ddo = resolve_did(algorithm_did)
    alg_datatoken = ocean.get_data_token(alg_ddo.data_token_address)
    return alg_ddo, alg_datatoken


def start_job(data_did, alg_did, wallet, model_id):
    ocean, _ = get_ocean()
    # make sure we operate on the updated and indexed metadata_cache_uri versions
    data_ddo = ocean.assets.resolve(data_did)
    alg_ddo = ocean.assets.resolve(alg_did)
    alg_datatoken = ocean.get_data_token(alg_ddo.data_token_address)

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
