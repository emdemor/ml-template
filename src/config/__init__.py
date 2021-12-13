import yaml


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

    features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "active", True)
        for params in features_config
    }


def get_feature_names(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return [elem["name"] for elem in features_config]


def get_feature_limits(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "limits", [None, None])
        for params in features_config
    }


def get_feature_transformations(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "transformation", "identity")
        for params in features_config
    }


def get_feature_types(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "type", "float")
        for params in features_config
    }


def get_feature_imputation_strategy(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "imputation_strategy", "mean")
        for params in features_config
    }


def get_feature_imputation_params(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "imputation_param", None)
        for params in features_config
    }


def get_feature_scalers(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "scaler", None)
        for params in features_config
    }


def get_feature_weights(features_config, only_actives=True):

    if only_actives:
        features_config = list(filter(lambda x: x["active"], features_config))

    return {
        params["name"]: set_default_values(params, "weight", 1)
        for params in features_config
    }