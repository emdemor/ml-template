import pandas as pd

from src.base.commons import to_snake_case
from src.base.file import read_file_string, download_file


def make_dataset(config, download_bases=False):

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

    return data