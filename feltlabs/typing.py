"""Module for special FELT types."""
from pathlib import Path
from typing import Any, Union

# Use this type for all models (TODO: better type)
Model = Any

# File type definiton
FileType = Union[str, Path, bytes]
PathType = Union[str, Path]
