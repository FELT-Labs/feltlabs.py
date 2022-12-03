"""Module for handling JSON files with numpy arrays."""
import base64
import json
from io import BytesIO
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from feltlabs.typing import FileType


def _numpy_to_str(array: Union[NDArray, np.integer, np.floating]) -> str:
    """Convert numpy array to compressed str."""
    bytes_io = BytesIO()
    np.savez_compressed(bytes_io, arr=array)
    enc = base64.b85encode(bytes_io.getvalue())
    return enc.decode()


def _str_to_numpy(data_str: str) -> NDArray:
    """Load numpy array from compressed str."""
    data = base64.b85decode(data_str)
    bytes_io = BytesIO(data)
    # allow_pickle must be False to prevent code execution
    array = np.load(bytes_io, allow_pickle=False)
    return array["arr"]


class NumpyEncoder(json.JSONEncoder):
    """Special class for handling numpy arrays in JSON."""

    # TODO: Remove uncompressed version? But compressed might be useful for some visualisations
    compress = True

    def default(self, obj: Any) -> Any:
        """Special method for encoding numpy arrays into JSON."""
        if (
            isinstance(obj, np.integer)
            or isinstance(obj, np.floating)
            or isinstance(obj, np.ndarray)
        ):
            return {
                "__numpy__": True,
                "array": _numpy_to_str(obj) if self.compress else obj.tolist(),
                "dtype": str(obj.dtype),
                "compress": self.compress,
            }

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
            if dct.get("compress", False):
                return _str_to_numpy(dct["array"])
            return np.array(dct["array"], dtype=dct["dtype"])
        return dct


def json_dump(obj: Any) -> bytes:
    """Convert object to JSON file represented by bytes."""
    return bytes(json.dumps(obj, cls=NumpyEncoder), "utf-8")


def json_load(file: Union[FileType, dict]) -> Any:
    """Load json file using custom loaded.

    Args:
        file: path to json file containing model produced by FELT labs.

    Returns:
        loaded json object (dictionary)
    """
    if isinstance(file, dict):
        return file

    if isinstance(file, bytes):
        return json.loads(file, object_hook=NumpyEncoder.decoder_hook)

    with open(file, "r") as f:
        return json.load(f, object_hook=NumpyEncoder.decoder_hook)
