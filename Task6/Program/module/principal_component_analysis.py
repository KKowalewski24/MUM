from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

# TODO ADD REAL PARAMS
COMPONENTS: List[float] = np.arange(2, 10, 1)
SVD_SOLVERS: List[str] = ["auto", "full", "arpack"]


def principal_component_analysis(
        datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        save_latex: bool = False) -> None:
    for ds_name in datasets:
        X_train, X_test, y_train, y_test = datasets[ds_name]["original"]
        for svd_solver in SVD_SOLVERS:
            for component in COMPONENTS:
                pca: PCA = PCA(n_components=component, svd_solver=svd_solver, random_state=21)
                pca.fit(X_train)
                X_train_transformed = pca.transform(X_train)
                X_test_transformed = pca.transform(X_test)
                key: str = "pca_" + svd_solver + "_" + str(component)
                datasets[ds_name][key] = (X_train_transformed, X_test_transformed, y_train, y_test)
