"""Function specific for Ocean's compute-to-data environment."""
import json
import os
from pathlib import Path

OUTPUT_FOLDER = Path("/data/outputs")
INPUT_FOLDER = Path("/data/inputs/")
CUSTOM_DATA = INPUT_FOLDER / "algoCustomData.json"


def get_ocean_config() -> dict[str, str]:
    with CUSTOM_DATA.open() as f:
        return json.load(f)


def get_dataset_files() -> list[Path]:
    """Get all file paths provided in Ocean's compute job environment.

    Returns:
        list of file paths
    """
    files = []
    dids = json.loads(os.getenv("DIDS", "[]"))
    for did in dids:
        # In future we might need to do different actions based on DID
        # just list all files in DID folder for now
        files.extend(list(INPUT_FOLDER.joinpath(did).glob("*")))
    return files


def save_output(name: str, data: bytes) -> None:
    """Save data to output folder by given name.

    Args:
        name: name of file to write
        data: data as bytes to store in the file
    """
    with open(OUTPUT_FOLDER / name, "wb+") as f:
        f.write(data)
