from os import error
import pandas as pd
import numpy as np

from src.base.commons import to_snake_case
from src.base.file import read_file_string, download_file
from src.model import __version__


def make_dataset(config, download_bases=False):

    data = pd.read_csv(config["data_local_path"])

    X = data.drop(columns="Drug", errors="ignore")

    y = data["Drug"]

    return X, y