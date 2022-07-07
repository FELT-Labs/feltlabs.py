"""Module for storing and managing data files."""
from typing import Any, Optional

from feltlabs.core import sklearn_to_json
from feltlabs.core.cryptography import encrypt_nacl
from feltlabs.typing import FileType, Model, PathType


def load_model(filename: FileType) -> Model:
    """Abstraction function for loading models.

    Args:
        filename: filepath or file-like object to load

    Returns:
        model object
    """
    return sklearn_to_json.import_model(filename)


def export_model(model: Model, path: Optional[PathType] = None) -> bytes:
    """Abstraction function for exporting model to file."""
    return sklearn_to_json.export_model(model, path)


def encrypt_model(model: Any, public_key: bytes) -> bytes:
    """Encrypt final model using provided public key.

    Args:
        model: scikit-learn trained model
        public_key: public key to use for encryption

    Returns:
        encrypted model as bytes
    """
    model_bytes = export_model(model)
    encrypted_model = encrypt_nacl(public_key, model_bytes)
    return encrypted_model
