from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

# Fractions from 0.2 to 0.9
COMPONENTS_FRACTIONS: List[float] = np.arange(0.2, 1.0, 0.1).round(2)
SVD_ALGORITHMS: List[str] = ["arpack", "randomized"]


def singular_value_decomposition(
        datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        save_latex: bool = False) -> None:
    for ds_name in datasets:
        X_train, X_test, y_train, y_test = datasets[ds_name]["original"]
        print(ds_name)

        min_dimension = min(X_train.shape[0], X_train.shape[1])

        for svd_algorithm in SVD_ALGORITHMS:
            print(svd_algorithm)
            for components_fraction in COMPONENTS_FRACTIONS:
                print(components_fraction)
                components_value = int((min_dimension * components_fraction))
                svd: TruncatedSVD = TruncatedSVD(n_components=components_value,
                                                 algorithm=svd_algorithm, random_state=21)
                svd.fit(X_train.astype(float))
                X_train_transformed = svd.transform(X_train)
                X_test_transformed = svd.transform(X_test)
                key: str = "svd_" + svd_algorithm + "_" + str(components_value)
                datasets[ds_name][key] = (X_train_transformed, X_test_transformed, y_train, y_test)
