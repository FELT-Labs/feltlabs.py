"""Module for storing and managing data files."""
import json
from typing import Any, Optional

from feltlabs.core.cryptography import encrypt_nacl
from feltlabs.core.models import analytics_model, sklearn_model
from feltlabs.typing import BaseModel, FileType


def load_model(file: FileType) -> BaseModel:
    """Load model from json file (intended for use in 3rd party programs).

    Args:
        file: path to json file containing model produced by FELT labs.

    Returns:
        scikit-learn model
    """
    if type(file) is bytes:
        data = json.loads(file)
    else:
        with open(file, "r") as f:
            data = json.load(f)

    if data["model_type"] == "sklearn":
        return sklearn_model.Model(data)
    elif data["model_type"] == "analytics":
        return analytics_model.Model(data)
    raise Exception("Invalid model type.")


def encrypt_model(model: BaseModel, public_key: bytes) -> bytes:
    """Encrypt final model using provided public key.

    Args:
        model: scikit-learn trained model
        public_key: public key to use for encryption

    Returns:
        encrypted model as bytes
    """
    model_bytes = model.export_model()
    encrypted_model = encrypt_nacl(public_key, model_bytes)
    return encrypted_model
