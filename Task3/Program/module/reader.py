from typing import Dict, List

import pandas as pd

from model.DividedDataSet import DividedDataSet


def read_csv_data_sets(filenames: List[str],
                       test_set_sizes: List[float]) -> Dict[int, List[DividedDataSet]]:
    data_sets: Dict[int, List[DividedDataSet]] = {}

    for i in range(len(filenames)):
        data_set = pd.read_csv(filenames[i])
        divided_data_sets: List[DividedDataSet] = []

        for test_set_size in test_set_sizes:
            # TODO ADD SPLITTING DATA SETS
            divided_data_sets.append(DividedDataSet(test_set_size, data_set, data_set))

        data_sets[i] = divided_data_sets

    return data_sets
