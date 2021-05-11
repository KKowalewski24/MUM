from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "knn"
K_RANGE = range(1, 30)
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def knn_classification(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       is_euclidean_metric: bool, data_set_name: str,
                       save_latex: bool = False) -> None:
    accuracy_list: List[List[float]] = []

    for k_value in K_RANGE:
        knn_classifier = KNeighborsClassifier(
            n_neighbors=k_value, p=2 if is_euclidean_metric else 1
        )
        knn_classifier.fit(data_set[0], data_set[2])
        y_prediction = knn_classifier.predict(data_set[1])
        accuracy = round(metrics.accuracy_score(data_set[3], y_prediction), 4)
        accuracy_list.append([accuracy])
        print("K parameter value: " + str(k_value) + ",\t" + "accuracy: " + str(accuracy))

    plt.plot(K_RANGE, accuracy_list)
    plt.ylabel("Accuracy")
    plt.xlabel("K parameter value")

    if save_latex:
        filename_description = "_" + data_set_name + ("_eucl" if is_euclidean_metric else "_manh")
        latex_generator.generate_horizontal_table(
            ["Accuracy"], list(K_RANGE), accuracy_list,
            "knn_table" + filename_description
        )
        image_filename = filename_description + "-" + datetime.now().strftime("%H%M%S")
        latex_generator.generate_chart_image("knn_chart" + image_filename)
        plt.savefig(LATEX_RESULTS_DIR + "/knn_chart" + image_filename)
        plt.close()

    plt.show()
