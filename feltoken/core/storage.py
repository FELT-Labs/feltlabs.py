"""Module for storing and managing data files at IPFS/Filecoin using web3.storage."""
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Union

import joblib

from feltoken.core.cryptography import encrypt_nacl

# File type definiton
FileType = Union[str, Path, BinaryIO]
PathType = Union[str, Path]


def load_model(filename: FileType) -> Any:
    """Abstraction function for loading models.

    Args:
        filename: filepath or file-like object to load

    Returns:
        model object
    """
    return joblib.load(filename)


def export_model(model: Any, path: PathType):
    """Abstraction function for exporting model to file."""
    joblib.dump(model, path)


def bytes_to_model(data: bytes):
    """Transform bytes into model."""
    return load_model(BytesIO(data))


def model_to_bytes(model: Any, path: PathType = "/tmp/model") -> bytes:
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


def encrypt_model(model: Any, public_key: bytes) -> bytes:
    """Encrypt final model using provided public key.

    Args:
        model: scikit-learn trained model
        public_key: public key to use for encryption

    Returns:
        encrypted model as bytes
    """
    model_bytes = model_to_bytes(model)
    encrypted_model = encrypt_nacl(public_key, model_bytes)
    return encrypted_model
