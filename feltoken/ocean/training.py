import base64
import os
import time
from io import BytesIO

from sklearn.linear_model import LinearRegression

from feltoken.core.storage import load_model, model_to_bytes
from feltoken.ocean.files import upload_model
from feltoken.ocean.ocean import get_ocean, get_wallet, start_job


def ocean_train_model(model, data_did, config):
    ocean, _ = get_ocean()
    wallet = get_wallet(config.account)
    print(data_did)

    # Get some random id
    model_id = base64.b64encode(os.urandom(16)).decode("ascii")

    model_bytes = model_to_bytes(model)
    upload_model(model_id, model_bytes)

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
    print(result)
    print(BytesIO(result))
    return load_model(BytesIO(result))


if __name__ == "__main__":
    # Just testing run:
    model = LinearRegression()
    ocean_train_model(model, None, None)
