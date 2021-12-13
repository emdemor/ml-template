import numpy as np
import pandas as pd
from src.config import *
from src.base.commons import dataframe_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline


class PreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_config):
        self.features_config = features_config
        self.feature_names = get_feature_names(self.features_config)
        self.feature_status = get_feature_status(self.features_config)
        self.feature_limits = get_feature_limits(self.features_config)
        self.feature_transformations = get_feature_transformations(self.features_config)
        self.feature_types = get_feature_types(self.features_config)
        self.feature_imputation_strategy = get_feature_imputation_strategy(
            self.features_config
        )
        self.feature_imputation_params = get_feature_imputation_params(
            self.features_config
        )
        self.feature_scalers = get_feature_scalers(self.features_config)
        self.feature_weights = get_feature_weights(self.features_config)

        self.preprocessor = make_pipeline(
            FeatureImputer(
                self.feature_names,
                self.feature_imputation_strategy,
                self.feature_imputation_params,
            ),
            FeatureClipper(self.feature_limits),
            FeatureTransformer(self.feature_transformations),
            FeatureScaler(self.feature_scalers),
            FeatureWeigher(self.feature_weights),
        )

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X, y=None):
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


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


class FeatureClipper(BaseEstimator, TransformerMixin):
    def __init__(self, features_limits):
        self.features_limits = features_limits

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for feature in self.features_limits:
            X[feature] = X[feature].clip(*self.features_limits[feature])
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features_transformations):
        self.features_transformations = features_transformations
        self.transformer = {
            feature: self.__interpret_transformation(features_transformations[feature])
            for feature in features_transformations
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for feature in self.features_transformations:
            X[feature] = X[feature].apply(self.transformer[feature])
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


class FeatureImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self, feature_names, feature_imputation_strategy, feature_imputation_params
    ):
        self.feature_names = feature_names
        self.feature_imputation_strategy = feature_imputation_strategy
        self.feature_imputation_params = feature_imputation_params
        self.imputer = ColumnTransformer(
            [
                (
                    feature,
                    self.__interpret_imputation(
                        self.feature_imputation_strategy[feature],
                        self.feature_imputation_params[feature],
                    ),
                    [i],
                )
                for i, feature in enumerate(self.feature_names)
            ]
        )

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
    def __init__(self, feature_scalers):
        self.feature_scalers = feature_scalers

        self.scaler = ColumnTransformer(
            [
                (
                    feature,
                    self.__interpret_scaler(self.feature_scalers[feature]),
                    [i],
                )
                for i, feature in enumerate(self.feature_scalers)
            ]
        )

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
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        for col in X.columns:
            X[col] = X[col] * self.feature_weights[col]

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
