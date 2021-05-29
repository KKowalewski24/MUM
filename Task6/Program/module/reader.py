from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE_VALUE = 21
TEST_DATA_PERCENTAGE = 0.3


def read_student_alcohol_consumption() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/student_alcohol_consumption.csv")
    _encode_columns(
        df, ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
             "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
             "internet", "romantic"]
    )
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return train_test_split(X, y, test_size=TEST_DATA_PERCENTAGE, random_state=RANDOM_STATE_VALUE)


def read_company_bankruptcy_prediction() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/company_bankruptcy_prediction.csv")
    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return train_test_split(X, y, test_size=TEST_DATA_PERCENTAGE, random_state=RANDOM_STATE_VALUE)


def read_wafer_manufacturing_anomalies() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/wafer_manufacturing_anomalies.csv")
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return train_test_split(X, y, test_size=TEST_DATA_PERCENTAGE, random_state=RANDOM_STATE_VALUE)


def _encode_columns(df: pd.DataFrame, columns_to_encode: List[str]) -> None:
    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        df[column] = label_encoder.fit_transform(df[column])
