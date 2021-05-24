from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


def read_iris_ds() -> np.ndarray:
    return pd.read_csv("data/Iris.csv").iloc[:, 1:6].to_numpy()


def read_mall_customers() -> np.ndarray:
    df = pd.read_csv("data/Mall_Customers.csv")
    df.iloc[:, 1] = LabelEncoder().fit_transform(df.iloc[:, 1])
    return df.iloc[:, 1:5].to_numpy()


# Returns only generated samples without class membership - in order
# to do this return tuple of samples and classes
def read_moons_ds() -> Tuple[np.ndarray, np.ndarray]:
    samples, classes = datasets.make_moons(n_samples=700, noise=0.09, random_state=1)
    return samples, classes
