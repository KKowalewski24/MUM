import numpy as np
import pandas as pd
from typing import List


def _insert_missing_values(df: pd.DataFrame, percent: int) -> pd.DataFrame:
    global_indices = np.random.permutation(df.size)[:df.size * percent // 100]
    for x, y in zip(global_indices // df.shape[1],
                    global_indices % df.shape[1]):
        df.iloc[x, y] = np.nan
    return df


def read(percents_of_missing_values: List[int]) -> List[pd.DataFrame]:
    df = pd.read_csv('../heart.csv')
    return [
        _insert_missing_values(df.copy(), percent)
        for percent in percents_of_missing_values
    ]
