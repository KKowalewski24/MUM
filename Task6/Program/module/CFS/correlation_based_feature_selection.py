from typing import Dict, Tuple

import numpy as np

from module.CFS.CFS import cfs


def correlation_based_feature_selection(
        datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        save_latex: bool = False) -> None:
    for ds_name in datasets:
        X_train, X_test, y_train, y_test = datasets[ds_name]["original"]
        print("ds_name: ", ds_name)
        d = X_train.shape[1]
        idx = cfs(X_train, y_train, d)
        for n in [.01, .05, .1, .3, .5, .75]:
            indexes = idx[:int(n * d)]
            X_train_transformed = X_train[:, indexes]
            X_test_transformed = X_test[:, indexes]

            key: str = "cfs_" + str(n) + "_n"
            datasets[ds_name][key] = (X_train_transformed, X_test_transformed, y_train, y_test)

# WYLICZONE wartosci indexow dla:
# ds_name:  letters
# 461, 471, 418, 460, 7, 358, 384, 452, 578, 470
#
# ds_name:  numerals
# 357, 269, 356, 418, 514, 576, 358, 484, 409, 434, 530, 509, 593, 490, 457, 107, 433, 560, 499, 429
#
# ds_name:  documents
# 545, 386, 201, 704, 518, 420, 831, 190, 606, 210, 206, 517, 813, 337, 594, 402, 630, 730, 672, 486
