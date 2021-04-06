import pandas as pd
from sklearn.impute import KNNImputer


def mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean())


def interpolate(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate().dropna()


def hot_deck(df: pd.DataFrame) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=1)
    return pd.DataFrame(imputer.fit_transform(df))


def regression(df: pd.DataFrame) -> pd.DataFrame:
    return df
