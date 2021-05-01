from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "db_scan"
MIN_SAMPLES_RANGE = np.arange(2, 10, 1)
EPSILON_RANGE = np.arange(0.1, 10, 0.1)

latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def db_scan_clustering(data_set: np.ndarray, data_set_name: str,
                       is_euclidean_metric: bool, save_latex: bool = False) -> None:
    # Tuple[List[epsilon], List[silhouette_score]]
    silhouette_scores: List[Tuple[List[float], List[float]]] = []

    for i in range(len(MIN_SAMPLES_RANGE)):
        silhouette_scores.append(([], []))
        for epsilon in EPSILON_RANGE:
            db_scan: DBSCAN = DBSCAN(
                min_samples=MIN_SAMPLES_RANGE[i], eps=epsilon, p=2 if is_euclidean_metric else 1
            )
            cluster_labels = db_scan.fit_predict(data_set)

            is_all_items_same = np.all(cluster_labels == cluster_labels[0])
            if not is_all_items_same:
                score = round(silhouette_score(data_set, cluster_labels), 4)
                silhouette_scores[i][0].append(epsilon)
                silhouette_scores[i][1].append(score)
                print("Silhouette:\t" + str(score))

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        data_set_name + (", Euclidean" if is_euclidean_metric else ", Manhattan") + " metric")

    for ax in axs.flat:
        ax.set(xlabel="Epsilon", ylabel="Silhouette Score")

    set_subplot(axs, 0, 0, silhouette_scores[0], MIN_SAMPLES_RANGE[0])
    set_subplot(axs, 0, 1, silhouette_scores[1], MIN_SAMPLES_RANGE[1])
    set_subplot(axs, 1, 0, silhouette_scores[2], MIN_SAMPLES_RANGE[2])
    set_subplot(axs, 1, 1, silhouette_scores[3], MIN_SAMPLES_RANGE[3])
    plt.show()

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        data_set_name + (", Euclidean" if is_euclidean_metric else ", Manhattan") + " metric")

    for ax in axs.flat:
        ax.set(xlabel="Epsilon", ylabel="Silhouette Score")
    set_subplot(axs, 0, 0, silhouette_scores[4], MIN_SAMPLES_RANGE[4])
    set_subplot(axs, 0, 1, silhouette_scores[5], MIN_SAMPLES_RANGE[5])
    set_subplot(axs, 1, 0, silhouette_scores[6], MIN_SAMPLES_RANGE[6])
    set_subplot(axs, 1, 1, silhouette_scores[7], MIN_SAMPLES_RANGE[7])
    plt.show()


def set_subplot(axs, row: int, column: int,
                score: Tuple[List[float], List[float]], min_sample: float) -> None:
    axs[row, column].plot(score[0], score[1])
    axs[row, column].set_title("min sample: " + str(min_sample))
