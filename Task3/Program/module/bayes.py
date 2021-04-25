from typing import List, Tuple
from sklearn import naive_bayes
from sklearn import metrics
import matplotlib.pyplot as plt

import numpy as np

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "bayes"
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def bayes_classification(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                         save_latex: bool, test_percentage: float) -> None:
    accuracy_list: List[List[float]] = []

    bayes_classifier = naive_bayes.GaussianNB()
    bayes_classifier.fit(data_set[0], data_set[2])
    y_prediction = bayes_classifier.predict(data_set[1])
    accuracy = round(metrics.accuracy_score(data_set[3], y_prediction), 4)
    accuracy_list.append([accuracy])
    print("Test data percentage: " + str(round(test_percentage * 100, 2)) + "% ,\t" + "accuracy: " + str(accuracy))

