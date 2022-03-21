"""Module for uploading files to data provider."""
import os
from io import BytesIO

import requests


def upload_model(_id: str, model_bytes: bytes) -> bool:
    """Upload file to data provider server.

    Args:
        _id: unique id of model file
        model_bytes: model object converted to bytes

    Return: true/false depending on success of upload.
    """
    url = f"{os.getenv('FILE_PROVIDER_URL')}/upload_model?_id={_id}"
    files = {"file": BytesIO(model_bytes)}
    r = requests.post(url, files=files)
    return r.json()["Status"] == "OK"
