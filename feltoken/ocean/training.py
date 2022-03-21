import base64
import os
import time
from io import BytesIO
from pathlib import Path

import joblib
from ocean_lib.assets.asset_resolver import resolve_asset
from ocean_lib.assets.trusted_algorithms import add_publisher_trusted_algorithm
from ocean_lib.web3_internal.wallet import Wallet
from sklearn.linear_model import LinearRegression

from feltoken.core.storage import load_model, model_to_bytes
from feltoken.ocean.dataset import publish_dataset
from feltoken.ocean.files import upload_model
from feltoken.ocean.ocean import get_ocean, publish_algorithm, start_job

LOGS = Path(__file__).parent
LOGS.mkdir(exist_ok=True)

ocean, provider_url = get_ocean()

wallet = Wallet(
    ocean.web3,
    os.getenv("TEST_PRIVATE_KEY1"),
    ocean.config.block_confirmations,
    ocean.config.transaction_timeout,
)

data_ddo = publish_dataset(wallet)


def get_algorithm(wallet):
    """Get already deployed algorithm or publish new."""
    # Load algorithm DDO or publish alg
    if (LOGS / "algorithm_ddo").exists():
        alg_ddo = joblib.load(LOGS / "algorithm_ddo")
        alg_datatoken = ocean.get_data_token(joblib.load(LOGS / "algorithm_token"))
    else:
        alg_ddo, alg_datatoken = publish_algorithm(wallet)
        joblib.dump(alg_ddo, LOGS / "algorithm_ddo")
        joblib.dump(alg_datatoken.address, LOGS / "algorithm_token")
    return alg_ddo, alg_datatoken


### ALLOW ALG FOR DATASET ###
# THIS WILL PROBABLY DONE SOMEHOW SEPARATELY, maybe do alg publishing through FELToken wallet?
# This waiting is necessary, or else ocean thing doesn't get updated and asset is
#  resolved to None which breaks the add_publisher_trusted_algorithm(...) function
for i in range(4):
    algo_ddo = resolve_asset(
        alg_ddo.did, metadata_cache_uri=ocean.config.metadata_cache_uri
    )
    if algo_ddo:
        break
    time.sleep(4)
add_publisher_trusted_algorithm(data_ddo, alg_ddo.did, ocean.config.metadata_cache_uri)
ocean.assets.update(data_ddo, publisher_wallet=wallet)


def train_model(model, data=None):
    # Get some random id
    model_id = base64.b64encode(os.urandom(16)).decode("ascii")

    model_bytes = model_to_bytes(model)
    upload_model(model_id, model_bytes)

    job_id = start_job(data_ddo.did, alg_ddo.did, alg_datatoken, wallet, model_id)

    ### Bob monitors logs / algorithm output ###
    for i in range(10):
        status = ocean.compute.status(data_ddo.did, job_id, wallet)
        print("Status", status)
        if status["status"] == 70:
            break
        time.sleep(10)

    result = ocean.compute.result_file(data_ddo.did, job_id, 0, wallet)
    print("result", result)
    return load_model(BytesIO(result))


if __name__ == "__main__":
    # Just testing run:
    model = LinearRegression()
    train_model(model, None)
