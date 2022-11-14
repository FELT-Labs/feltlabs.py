"""Module for handling JSON files with numpy arrays."""
import json
from typing import Any

import numpy as np

from feltlabs.typing import FileType


class NumpyEncoder(json.JSONEncoder):
    """Special class for handling numpy arrays in JSON."""

    def default(self, obj: Any) -> Any:
        """Special method for encoding numpy arrays into JSON."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {"__numpy__": True, "array": obj.tolist(), "dtype": str(obj.dtype)}

        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def decoder_hook(dct: dict) -> Any:
        """Static method for decoding numpy arrays from JSON.

        Args:
            dct: dictionary or list of loaded json

        Returns:
            new dictionary with decoded numpy arrays
        """
        if "__numpy__" in dct:
            return np.array(dct["array"], dtype=dct["dtype"])
        return dct


def json_dump(obj: Any) -> bytes:
    """Convert object to JSON file represented by bytes."""
    return bytes(json.dumps(obj, cls=NumpyEncoder), "utf-8")


def json_load(file: FileType) -> Any:
    """Load json file using custom loaded.

    Args:
        file: path to json file containing model produced by FELT labs.

    Returns:
        loaded json object (dictionary)
    """
    if type(file) is bytes:
        return json.loads(file, object_hook=NumpyEncoder.decoder_hook)

    with open(file, "r") as f:
        return json.load(f, object_hook=NumpyEncoder.decoder_hook)
