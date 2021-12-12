import re
import yaml


def to_snake_case(name):

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)

    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    name = name.replace("__", "_")

    return name


def get_config(filename):

    with open(filename, "r") as file:
        settings = yaml.safe_load(file)

    return settings
