from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data_set_from_csv_file(filename: str, y_column: str, test_set_size: float = 0.3) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(filename)
    X = df.drop(y_column, axis=1).to_numpy()
    y = df[y_column].to_numpy()
    return train_test_split(X, y, test_size=test_set_size)
