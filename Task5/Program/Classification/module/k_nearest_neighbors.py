from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from module.LatexGenerator import LatexGenerator

K_RANGE = range(1, 30)
METRICS: Dict[int, str] = {1: "Manhattan", 2: "Euclidean"}
LATEX_RESULTS_DIR = "knn"
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def knn_classification(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       data_set_name: str, save_latex: bool = False) -> None:
    X_train, X_test, y_train, y_test = data_set

    plt.suptitle("dataset: " + data_set_name)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for k_value in K_RANGE:
        for metric in reversed(METRICS):
            knn_classifier = KNeighborsClassifier(
                n_neighbors=k_value, p=metric
            )
            knn_classifier.fit(X_train, y_train)
            y_prediction = knn_classifier.predict(X_test)
            accuracy = round(metrics.accuracy_score(y_test, y_prediction), 4)
            print(
                "Metric:", METRICS[metric], "\t",
                "K parameter value:", k_value,
                "\t", "accuracy:", accuracy
            )

    # plt.show()
