"""Module for executing model training."""
import argparse
from typing import Any, Optional, Union, cast

import numpy as np

from feltoken.core.data import load_data
from feltoken.core.storage import export_model, load_model
from feltoken.node.config import Config


class TrainingConfig:
    model: str
    data: str
    output_model: str
    account: Optional[str]


# TODO: type for the model
def train_model(
    model: Any,
    data: Union[tuple[np.ndarray, np.ndarray], str],
    config: Union[TrainingConfig, Config],
) -> Any:
    """Universal model training function, it starts training depending on selected type.

    Args:
        model: initial model object with fit() method
        data: tuple[X, Y] or string depending on training type
        config: object specifing traing type

    Returns:
        new model object with trained weights
    """
    assert type(data) == tuple, "Invalid data type"
    model.fit(data[0], data[1])
    return model


def parse_args(args_str: Optional[str] = None) -> TrainingConfig:
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str: string with arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Script for training models, possible to execute from command line."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model file which should be loaded and trained.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help="Path to CSV file with data. Last column is considered as Y.",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        help="Path to store the final model file.",
    )
    args = parser.parse_args(args_str)

    return cast(TrainingConfig, args)


def main():
    """Main function for exectuting from command line."""
    args = parse_args()
    model = load_model(args.model)

    data = load_data(args)

    model = train_model(model, data, args)
    export_model(model, args.output_model)


if __name__ == "__main__":
    main()
