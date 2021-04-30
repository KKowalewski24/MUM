import pandas as pd


def read_iris_ds() -> pd.DataFrame:
    return pd.read_csv('data/Iris.csv').iloc[:, 1:5]
