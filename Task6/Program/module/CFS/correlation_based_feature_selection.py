from typing import Dict, Tuple
from module.CFS.CFS import cfs

import numpy as np


def correlation_based_feature_selection(
        datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        save_latex: bool = False) -> None:
    for ds_name in datasets:
        X_train, X_test, y_train, y_test = datasets[ds_name]["original"]
        print("ds_name: ", ds_name)
        idx = cfs(X_train, y_train, 2)
        X_train_transformed = []
        Y_train_transformed = []
        X_test_transformed = []
        Y_test_transformed = []
        for index in idx:
            X_train_transformed.append(X_train[index])
            Y_train_transformed.append(y_train[index])
            X_test_transformed.append(X_test[index])
            Y_test_transformed.append(y_test[index])

        key: str = "cfs_" #+ n_stop + "_n_stop"
        datasets[ds_name][key] = (X_train_transformed, X_test_transformed, Y_train_transformed, Y_test_transformed)
