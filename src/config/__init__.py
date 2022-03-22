import yaml
from io import StringIO


def get_config(filename):

    with open(filename, "r") as file:
        settings = yaml.safe_load(file)

    return settings


def set_default_values(dictionary, key, default):
    if key not in dictionary:
        return default
    else:
        return dictionary[key]


def get_feature_status(features_config):

    features_config = list(
        filter(lambda x: x["active"] if "active" in x else True, features_config)
    )

    return {
        params["name"]: set_default_values(params, "active", True)
        for params in features_config
    }


def get_feature_names(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return [elem["name"] for elem in features_config]


def get_feature_limits(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return {
        params["name"]: set_default_values(params, "limits", [None, None])
        for params in features_config
    }


def get_feature_transformations(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return {
        params["name"]: set_default_values(params, "transformation", "identity")
        for params in features_config
    }


def get_feature_types(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return {
        params["name"]: set_default_values(params, "type", "float")
        for params in features_config
    }


def get_feature_imputation_strategy(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return {
        params["name"]: set_default_values(params, "imputation_strategy", "mean")
        for params in features_config
    }


def get_feature_imputation_params(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return {
        params["name"]: set_default_values(params, "imputation_param", None)
        for params in features_config
    }


def get_feature_scalers(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return {
        params["name"]: set_default_values(params, "scaler", None)
        for params in features_config
    }


def get_feature_weights(features_config, only_actives=True):

    if only_actives:
        features_config = list(
            filter(lambda x: x["active"] if "active" in x else True, features_config)
        )

    return {
        params["name"]: set_default_values(params, "weight", 1)
        for params in features_config
    }


def get_column_type(series):
    dtype = series.dtype.name

    if "int" in str(dtype):
        dtype = "int"

    return dtype


def column_properties(series):

    results = dict(
        name=series.name,
        active=True,
        type=get_column_type(series),
        limits="[left(]" + str([series.min(), series.max()]) + "[right]",
        transformation="identity",
        imputation_strategy="mean",
        imputation_param=None,
        scaler="min_max",
        weight=1,
    )

    return results


def init_config_file(dataframe, filename):

    data = [column_properties(dataframe[col]) for col in dataframe.columns]

    string_stream = StringIO()

    yaml.dump(data, string_stream, default_flow_style=False, sort_keys=False)

    main_string = string_stream.getvalue()

    string_stream.close()

    main_string = (
        main_string.replace("\n- name:", "\n\n- name:")
        .replace("'[left(]", "")
        .replace("[right]'", "")
    )

    with open(filename, "w") as outfile:
        outfile.write(main_string)
