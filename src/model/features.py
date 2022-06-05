import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def build_features(X: pd.DataFrame, y: pd.Series) -> tuple:

    X["Sex"] = X["Sex"].replace(
        {
            "F": 0,
            "M": 1,
        }
    )

    X["BP_LOW"] = (X["BP"] == "LOW").astype("Int8")
    X["BP_HIGH"] = (X["BP"] == "HIGH").astype("Int8")

    X["Cholesterol"] = X["Cholesterol"].replace(
        {
            "NORMAL": 0,
            "HIGH": 1,
        }
    )

    y = y.replace(
        {
            "drugA": 1,
            "drugB": 2,
            "drugC": 3,
            "drugX": 4,
            "DrugY": 5,
        }
    )

    return X, y