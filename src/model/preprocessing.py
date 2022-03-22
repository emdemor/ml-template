import numpy as np
import pandas as pd
import logging
from src.config import *
from src.base.commons import dataframe_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline

from sklearn import compose
from sklearn.pipeline import Pipeline


class Identity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class ColumnTransformer:
    def __init__(self, *args, **kwargs):
        self.column_transformer = compose.ColumnTransformer(*args, **kwargs)

    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X, y=None):
        return dataframe_transformer(X, self.column_transformer)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformation):
        self.transformation = transformation
        self.transformer = self.__interpret_transformation(self.transformation)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X.copy()

        try:
            X = X.apply(self.transformer)

        except Exception as err:
            logging.info(err)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __interpret_transformation(self, transformation):

        if transformation == "log":
            return np.log

        elif transformation == "log10":
            return np.log10

        elif transformation == "log1p":
            return np.log1p

        elif transformation == "exp":
            return np.exp

        elif transformation == "square":
            return np.square

        elif transformation == "sqrt":
            return np.sqrt

        elif transformation == "identity":
            return lambda x: x

        else:
            return lambda x: x


class FeatureClipper(BaseEstimator, TransformerMixin):
    def __init__(self, limits):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X.copy()

        try:
            X = X.clip(*self.limits)

        except Exception as err:
            logging.info(err)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class FeatureImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, parameter=None):
        self.strategy = strategy
        self.parameter = parameter
        self.imputer = self.__interpret_imputation(self.strategy, self.parameter)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        return dataframe_transformer(X, self.imputer)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __interpret_imputation(self, imputation, param):

        if imputation == "mean":
            return SimpleImputer(strategy="mean")

        elif imputation == "median":
            return SimpleImputer(strategy="median")

        elif imputation == "constant":
            return SimpleImputer(strategy="constant", fill_value=param)

        else:
            return Identity()


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy):
        self.strategy = strategy
        self.scaler = self.__interpret_scaler(self.strategy)

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        return dataframe_transformer(X, self.scaler)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __interpret_scaler(self, scaler):

        if scaler == "min_max":
            return MinMaxScaler()

        elif scaler == "standard":
            return StandardScaler()

        elif scaler == "robust":
            return RobustScaler()

        elif scaler == None:
            return Identity()

        else:
            return Identity()


class FeatureWeigher(BaseEstimator, TransformerMixin):
    def __init__(self, weight):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col] * self.weight

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class PreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_config):

        for config in features_config:
            if "active" not in config:
                config.update({"active": True})

        feature_names = [config["name"] for config in features_config]
        feature_active = {
            config["name"]: config["active"] for config in features_config
        }
        feature_types = {config["name"]: config["type"] for config in features_config}

        df = pd.DataFrame(
            [
                (j, config["name"], config["active"], config["type"], i, k, config[k])
                for j, config in enumerate(features_config)
                for i, k in enumerate(
                    {
                        k: config[k]
                        for k in filter(
                            lambda x: x
                            not in (
                                "name",
                                "active",
                                "type",
                                "encode",
                                "polynomial_degree",
                            ),
                            config.keys(),
                        )
                    }
                )
            ],
            columns=[
                "feature_order",
                "feature_name",
                "active",
                "type",
                "transform_order",
                "key",
                "value",
            ],
        )

        for order in df["transform_order"].unique():
            for col in feature_names:
                if (
                    len(
                        df[
                            (df["feature_name"] == col)
                            & (df["transform_order"] == order)
                        ]
                    )
                    == 0
                ):
                    df = df.append(
                        [
                            {
                                "feature_order": feature_names.index(col),
                                "feature_name": col,
                                "active": feature_active[col],
                                "type": feature_types[col],
                                "transform_order": order,
                                "key": "transformation",
                                "value": "identity",
                            }
                        ]
                    )

        df = df.sort_values(["transform_order", "feature_order"])

        steps = []

        for transform_order in df["transform_order"].unique():

            df_step = df[df["transform_order"] == transform_order].sort_values(
                "feature_order"
            )
            transformers = [
                (n, self.__interpret_process_step(key, value), [o])
                for n, o, key, value in df_step[
                    ["feature_name", "feature_order", "key", "value"]
                ].to_numpy()
            ]

            steps.append(
                (
                    f"step_{transform_order}",
                    ColumnTransformer(transformers=transformers),
                )
            )

        self.preprocessor = Pipeline(steps=steps)

    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X, y=None):
        return dataframe_transformer(X, self.preprocessor)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __interpret_process_step(self, key, value):
        if key == "transformation":
            return FeatureTransformer(transformation=value)

        elif key == "limits":
            return FeatureClipper(limits=value)

        elif key == "imputation_strategy":

            strategy, *param = str(value).split(":")

            if strategy == "constant":
                param = float(param[0])
            else:
                param = None

            return FeatureImputer(strategy="constant", parameter=param)

        elif key == "scaler":
            return FeatureScaler(strategy=value)

        elif key == "weight":
            return FeatureWeigher(weight=value)

        else:
            return Identity()
