"""Module for uploading files to data provider."""
import os
from io import BytesIO
from typing import Any

import requests

from feltoken.core.storage import model_to_bytes


def upload_model(_id: str, model: Any) -> bool:
    """Upload file to data provider server.

    Args:
        _id: unique id of model file
        model_bytes: model object converted to bytes

    Return: true/false depending on success of upload.
    """
    model_bytes = model_to_bytes(model)
    url = f"{os.getenv('FILE_PROVIDER_URL')}/upload_model?_id={_id}"
    files = {"file": BytesIO(model_bytes)}
    for _ in range(5):
        try:
            r = requests.post(
                url,
                files=files,
            )
            return r.json()["Status"] == "OK"
        except requests.exceptions.ConnectionError:
            print("Connection error - retry")
    return False
