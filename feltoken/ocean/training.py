"""Module for training using Ocean protocol compute-to-data."""
import base64
import os
import time
from io import BytesIO
from typing import Any, Union

from feltoken.core.storage import load_model
from feltoken.node.config import Config
from feltoken.node.training import TrainingConfig
from feltoken.ocean.files import upload_model
from feltoken.ocean.ocean import get_ocean, get_wallet, start_job


def ocean_train_model(
    model: Any, data_did: str, config: Union[TrainingConfig, Config]
) -> Any:
    """Train model using compute-to-data."""
    ocean, _ = get_ocean()
    wallet = get_wallet(config.account)

    # Get some random id
    model_id = base64.urlsafe_b64encode(os.urandom(16)).decode("ascii")

    upload_model(model_id, model)

    job_id = start_job(data_did, config.algorithm_did, wallet, model_id)

    # Monitor alg status until finished
    status = None
    for i in range(20):
        status = ocean.compute.status(data_did, job_id, wallet)
        print("Status", status)
        if status["status"] == 70:
            break
        time.sleep(10)

    assert status and status["status"] == 70, f"Training job didn't finish: {status}"

    result = ocean.compute.result_file(data_did, job_id, 0, wallet)
    return load_model(BytesIO(result))
