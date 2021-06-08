from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

# Fractions from 0.3 to 0.95
COMPONENTS_FRACTIONS: List[float] = np.arange(0.3, 1.0, 0.05).round(2)
SVD_SOLVERS: List[str] = ["auto", "full", "arpack", "randomized"]


def principal_component_analysis(
        datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        save_latex: bool = False) -> None:
    for ds_name in datasets:
        X_train, X_test, y_train, y_test = datasets[ds_name]["original"]

        # Find min in shape - rows and cols number - this is required
        # for proper working of PCA from sklearn
        min_dimension = min(X_train.shape[0], X_train.shape[1])

        for svd_solver in SVD_SOLVERS:
            for components_fraction in COMPONENTS_FRACTIONS:
                # Fraction multiplied by min dimension (number of cols or rows) in certain dataset
                components_value: int = round(components_fraction * min_dimension)
                pca: PCA = PCA(n_components=components_value,
                               svd_solver=svd_solver, random_state=21)
                pca.fit(X_train)
                X_train_transformed = pca.transform(X_train)
                X_test_transformed = pca.transform(X_test)
                key: str = "pca_" + svd_solver + "_" + str(components_value)
                datasets[ds_name][key] = (X_train_transformed, X_test_transformed, y_train, y_test)
