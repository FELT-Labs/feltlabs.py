"""Function specific for Ocean's compute-to-data environment."""
from pathlib import Path

from feltlabs.config import OceanConfig


def get_dataset_files(config: OceanConfig) -> list[Path]:
    """Get all file paths provided in Ocean's compute job environment.

    Args:
        config: ocean config containing output path

    Returns:
        list of file paths
    """
    files = []
    for path in config.input_folder.glob("**/*"):
        if path.is_file() and path.name != config.custom_data:
            files.append(path)
    return files


def save_output(name: str, data: bytes, config: OceanConfig) -> None:
    """Save data to output folder by given name.

    Args:
        name: name of file to write
        data: data as bytes to store in the file
        config: ocean config containing output path
    """
    with open(config.output_folder / name, "wb+") as f:
        f.write(data)
