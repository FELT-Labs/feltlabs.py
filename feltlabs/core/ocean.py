"""Function specific for Ocean's compute-to-data environment."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

from feltlabs.config import OceanConfig


class DatasetConfig:
    """Config class for loading datasets."""

    data_type: str
    load_code: str
    _files: Dict[int, Path]

    @property
    def files(self) -> List[Tuple[Path, int]]:
        """Get files as list sorted by index."""
        return sorted(
            [(path, index) for index, path in self._files.items()], key=lambda x: x[1]
        )

    def __init__(self):
        """Init config class with default values."""
        self.data_type = "default"
        self._files = {}
        self.load_code = ""

    def parse_config(self, path: Path) -> None:
        """Load FELT config file from path."""
        with path.open() as f:
            config = json.load(f)

        self.data_type = config.get("data_type", self.data_type)
        self.load_code = config.get("load_code", self.load_code)

    def add_file(self, path: Path) -> None:
        """Add data file path to dataset object."""
        index = int(path.name)
        self._files[index] = path


def _is_dataset_config(path: Path) -> bool:
    """Check if file is FELT dataset config file."""
    try:
        with path.open() as f:
            config = json.load(f)
            if config["name"] == "FELT_CONFIG":
                return True
    except Exception:
        pass
    return False


def get_datasets(config: OceanConfig) -> Dict[str, DatasetConfig]:
    """Get all dataset paths provided in Ocean's compute job environment.

    Args:
        config: ocean config containing output path

    Returns:
        dictionary mapping dataset DID to dataset loading config
    """
    did_folders = [(p, p.name) for p in config.input_folder.iterdir() if p.is_dir()]

    datasets = {}
    for folder, did in did_folders:
        datasets = {did: DatasetConfig()}

        for path in folder.glob("**/*"):
            if path.is_file() and _is_dataset_config(path):
                datasets[did].parse_config(path)
            elif path.is_file():
                datasets[did].add_file(path)

    return datasets


def save_output(name: str, data: bytes, config: OceanConfig) -> None:
    """Save data to output folder by given name.

    Args:
        name: name of file to write
        data: data as bytes to store in the file
        config: ocean config containing output path
    """
    with open(config.output_folder / name, "wb+") as f:
        f.write(data)
