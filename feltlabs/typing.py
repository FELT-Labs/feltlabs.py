"""Module for special FELT types."""
from pathlib import Path
from typing import Any, Union

# File type definiton
FileType = Union[str, Path, bytes]
PathType = Union[str, Path]

# Use this type for all models (TODO: better type)
# Keep this last to prevent circular import
from feltlabs.core.models.base_model import BaseModel

BaseModel = BaseModel
