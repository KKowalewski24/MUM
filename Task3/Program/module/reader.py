from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# Returns dict of set order number and tuple that contains training set and test set
def read_csv_data_sets(filenames: List[str],
                       training_set_size: int) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    data_sets: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for i in range(len(filenames)):
        data_set = pd.read_csv(filenames[i])
        train_test_split()
        data_sets[i] = (data_set, data_set)

    return data_sets
