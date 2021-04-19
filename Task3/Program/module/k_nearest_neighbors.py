from typing import Tuple

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from module.LatexGenerator import LatexGenerator

latex_generator: LatexGenerator = LatexGenerator("knn")

K_RANGE = range(1, 3)


def knn_classification(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       save_latex: bool) -> None:
    for k_number in K_RANGE:
        knn_classifier = KNeighborsClassifier(n_neighbors=k_number)
        knn_classifier.fit(data_set[0], data_set[2])
        y_prediction = knn_classifier.predict(data_set[1])
        print(metrics.accuracy_score(data_set[3], y_prediction))
