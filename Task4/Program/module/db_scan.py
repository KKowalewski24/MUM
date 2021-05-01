from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "db_scan"
MIN_SAMPLES_RANGE = np.arange(2, 10, 1)
EPSILON_RANGE = np.arange(0.1, 10, 0.1)
CHARTS_NUMBER = 4

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

    draw_chart(data_set_name, is_euclidean_metric, silhouette_scores, 0)
    draw_chart(data_set_name, is_euclidean_metric, silhouette_scores, 1)
    plt.show()


def draw_chart(data_set_name: str, is_euclidean_metric: bool,
               score: List[Tuple[List[float], List[float]]], order_number: int) -> None:
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        data_set_name + (", Euclidean" if is_euclidean_metric else ", Manhattan") + " metric"
    )

    for ax in axs.flat:
        ax.set(xlabel="Epsilon", ylabel="Silhouette Score")

    set_subplot(axs, 0, 0, score[0 + CHARTS_NUMBER * order_number],
                MIN_SAMPLES_RANGE[0 + CHARTS_NUMBER * order_number])
    set_subplot(axs, 0, 1, score[1 + CHARTS_NUMBER * order_number],
                MIN_SAMPLES_RANGE[1 + CHARTS_NUMBER * order_number])
    set_subplot(axs, 1, 0, score[2 + CHARTS_NUMBER * order_number],
                MIN_SAMPLES_RANGE[2 + CHARTS_NUMBER * order_number])
    set_subplot(axs, 1, 1, score[3 + CHARTS_NUMBER * order_number],
                MIN_SAMPLES_RANGE[3 + CHARTS_NUMBER * order_number])


def set_subplot(axs, row: int, column: int,
                score: Tuple[List[float], List[float]], min_sample: float) -> None:
    axs[row, column].plot(score[0], score[1])
    axs[row, column].set_title("min sample: " + str(min_sample))
