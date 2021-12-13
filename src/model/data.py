from os import error
import pandas as pd
import numpy as np

from src.base.commons import to_snake_case
from src.base.file import read_file_string, download_file
from src.model import __version__


def make_dataset(config, download_bases=False):

    targets = [
        "pastry",
        "z_scratch",
        "k_scatch",
        "stains",
        "dirtiness",
        "bumps",
        "other_faults",
    ]

    if download_bases:

        download_file(url=config["data_url"], destination=config["data_local_path"])

        download_file(
            url=config["columns_url"], destination=config["columns_local_path"]
        )

    columns = list(
        map(
            to_snake_case,
            read_file_string(config["columns_local_path"], splitlines=True),
        )
    )

    data = pd.read_csv(config["data_local_path"], sep="\t", header=None)

    data.columns = columns

    X = data.drop(columns=targets, errors="ignore")

    y = np.sum(data[targets] * np.arange(1, 1 + len(targets)).reshape(1, -1), axis=1)

    return X, y