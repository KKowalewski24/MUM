from typing import Dict, Tuple

import numpy as np


def variance_analysis(
        datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
        save_latex: bool = False) -> None:
    for ds_name in datasets:
        X_train, X_test, y_train, y_test = datasets[ds_name]["original"]
        va = PerClassVarianceAnalysisBasedFeatureSelection()
        va.fit(X_train, y_train)
        for i in [
            0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9
        ]:
            d = int(i * X_train.shape[1])
            datasets[ds_name]["va_" + str(i) + "_" + str(d)] = (va.transform(
                X_train, d), va.transform(X_test, d), y_train, y_test)


class PerClassVarianceAnalysisBasedFeatureSelection():

    def __init__(self):
        self.weights_ = None


    def fit(self, X, y):
        all_variance = np.var(X, axis=0)
        per_class_variances = []
        for c in np.unique(y):
            per_class_variances.append(np.var(X[y == c], axis=0))
        self.weights_ = all_variance / np.average(per_class_variances, axis=0)
        self.weights_[np.isnan(self.weights_)] = np.NINF


    def transform(self, X, n):
        return X[:, np.argsort(self.weights_)[-n:]]
