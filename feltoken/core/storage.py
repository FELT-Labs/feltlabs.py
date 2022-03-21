"""Module for storing and managing data files at IPFS/Filecoin using web3.storage."""
import os
import time

import httpx
import joblib

from feltoken.core.web3 import decrypt_bytes


def load_model(filename):
    """Abstraction function for loading models.

    Args:
        filename: filepath or file-like object to load

    Returns:
        model object
    """
    return joblib.load(filename)


def export_model(model, path):
    """Abstraction function for exporting model to file."""
    joblib.dump(model, path)


def model_to_bytes(model, path="/tmp/model") -> bytes:
    """Convert model to bytes which can be stored/exchanged/loaded.

    Args:
        model: model object
        path: path-like object where to store the model

    Return:
        bytes representing the model
    """
    # TODO: Optimize this part, so we don't need to r/w, json.dump to memory,
    #       But sometimes we actually want to store the model so it's ok so far
    export_model(model, path)
    with open(path, "rb") as f:
        return f.read()


def ipfs_upload_file(file):
    """Upload file to IPFS using web3.storage.

    Args:
        file: file-like object in byte mode.

    Returns:
        Response: httpx response object
    """
    # TODO: Check for upload error
    return httpx.post(
        "https://api.web3.storage/upload",
        headers={"Authorization": "Bearer " + os.environ["WEB3_STORAGE_TOKEN"]},
        files={"file": file},
        timeout=None,
    )


def ipfs_download_file(cid, output_path=None, secret=None):
    """Download file stored in IPFS.

    Args:
        cid (str): string describing location of the file.
        output_path (Optiona[str]): if set file will be stored at this path.

    Returns:
        Response: httpx response object
    """
    for _ in range(5):
        try:
            res = httpx.get(f"https://{cid}.ipfs.dweb.link/", timeout=10.0)
        except httpx.ReadTimeout:
            print("Connection timeout - retry")
            time.sleep(5)

    content = res.content
    if secret is not None:
        content = decrypt_bytes(res.content, secret)

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(content)

    return content
