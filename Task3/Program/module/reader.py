from typing import Dict, List

import pandas as pd


def read_csv_data_sets(filenames: List[str]) -> Dict[int, pd.DataFrame]:
    data_sets: Dict[int, pd.DataFrame] = {}
    for i in range(len(filenames)):
        data_sets[i] = pd.read_csv(filenames[i])

    return data_sets
