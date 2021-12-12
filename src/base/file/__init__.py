import requests


def read_file_string(filepath, splitlines=False):

    with open(filepath, "r") as file:
        data = file.read()

    if splitlines:
        data = data.splitlines()

    return data


def get_filename_from_url(url):
    return url.split("/")[-1]


def download_file(url, destination=None):

    if destination is None:
        destination = get_filename_from_url(url)

    r = requests.get(url, allow_redirects=True)

    with open(destination, "wb") as file:
        file.write(r.content)
