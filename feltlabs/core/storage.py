"""Module for storing and managing data files."""
from typing import Union

from feltlabs.core.cryptography import encrypt_nacl
from feltlabs.core.json_handler import json_load
from feltlabs.core.models import analytics_model, sklearn_model, tensorflow_model
from feltlabs.core.models.base_model import BaseModel
from feltlabs.typing import FileType


def _is_experimental(experimental: bool) -> None:
    """Raise exception if loading experimental model while experimental is False."""
    if not experimental:
        raise Exception("Loading experimental model while experimental is set False.")


def load_model(file: Union[FileType, dict], experimental: bool = True) -> BaseModel:
    """Load model from json file (intended for use in 3rd party programs).

    Args:
        file: path to json file containing model produced by FELT labs.
        experimental: allow loading experimental models (provides lower security)

    Returns:
        scikit-learn model
    """
    data = json_load(file)["model_definition"]

    if data["model_type"] == "sklearn":
        return sklearn_model.Model(data)
    elif data["model_type"] == "analytics":
        return analytics_model.Model(data)
    elif data["model_type"] == "tensorflow":
        _is_experimental(experimental)
        return tensorflow_model.Model(data)
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
