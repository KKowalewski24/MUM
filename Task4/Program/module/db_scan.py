from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "db_scan"
MIN_SAMPLES_RANGE = np.arange(2, 10, 1)
EPSILON_RANGE = np.arange(0.1, 10, 0.1).round(2)
SUBPLOTS_PER_CHART = 4
MIN_SAMPLE_PER_TABLE = 4

latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def db_scan_clustering(data_set: np.ndarray, data_set_name: str,
                       is_euclidean_metric: bool, save_latex: bool = False) -> None:
    epsilons_and_scores: List[Tuple[List[float], List[float]]] = []

    for i in range(len(MIN_SAMPLES_RANGE)):
        epsilons_and_scores.append(([], []))
        for epsilon in EPSILON_RANGE:
            db_scan: DBSCAN = DBSCAN(
                min_samples=MIN_SAMPLES_RANGE[i], eps=epsilon, p=2 if is_euclidean_metric else 1
            )
            cluster_labels = db_scan.fit_predict(data_set)

            is_all_items_same = np.all(cluster_labels == cluster_labels[0])
            if not is_all_items_same:
                score = round(silhouette_score(data_set, cluster_labels), 4)
                epsilons_and_scores[i][0].append(epsilon)
                epsilons_and_scores[i][1].append(score)

        _draw_score(epsilons_and_scores[i], MIN_SAMPLES_RANGE[i], data_set_name,
                    is_euclidean_metric, save_latex)
    _draw_chart(epsilons_and_scores, data_set_name, is_euclidean_metric, save_latex, 0)
    _draw_chart(epsilons_and_scores, data_set_name, is_euclidean_metric, save_latex, 1)


def _draw_score(epsilons_and_scores: Tuple[List[float], List[float]], min_sample: int,
                data_set_name: str, is_euclidean_metric: bool, save_latex: bool) -> None:
    print("Min Sample:\t" + str(min_sample))

    latex_scores: List[List[float]] = []
    for i in range(len(epsilons_and_scores[0])):
        epsilon = epsilons_and_scores[0][i]
        score = epsilons_and_scores[1][i]
        latex_scores.append([epsilon, score])
        print("\tEpsilon:\t" + str(epsilon), end=", ")
        print("\tSilhouette:\t" + str(score))

    if save_latex:
        latex_generator.generate_vertical_table(
            ["Epsilon", "Silhouette"], latex_scores,
            "db_scan_table_" + data_set_name + ("_eucl" if is_euclidean_metric else "_manh") \
            + "_min_sample" + str(min_sample)
        )


def _draw_chart(epsilons_and_scores: List[Tuple[List[float], List[float]]], data_set_name: str,
                is_euclidean_metric: bool, save_latex: bool, order_number: int) -> None:
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    fig.suptitle(
        data_set_name + (", Euclidean" if is_euclidean_metric else ", Manhattan") + " metric"
    )

    for ax in axs.flat:
        ax.set(xlabel="Epsilon", ylabel="Silhouette Score")

    _set_subplot(
        axs, 0, 0, epsilons_and_scores[0 + SUBPLOTS_PER_CHART * order_number],
        MIN_SAMPLES_RANGE[0 + SUBPLOTS_PER_CHART * order_number]
    )
    _set_subplot(
        axs, 0, 1, epsilons_and_scores[1 + SUBPLOTS_PER_CHART * order_number],
        MIN_SAMPLES_RANGE[1 + SUBPLOTS_PER_CHART * order_number]
    )
    _set_subplot(
        axs, 1, 0, epsilons_and_scores[2 + SUBPLOTS_PER_CHART * order_number],
        MIN_SAMPLES_RANGE[2 + SUBPLOTS_PER_CHART * order_number]
    )
    _set_subplot(
        axs, 1, 1, epsilons_and_scores[3 + SUBPLOTS_PER_CHART * order_number],
        MIN_SAMPLES_RANGE[3 + SUBPLOTS_PER_CHART * order_number]
    )

    if save_latex:
        base_filename = "_" + data_set_name + ("_eucl" if is_euclidean_metric else "_manh")
        image_filename = base_filename + str(order_number) + "-" \
                         + datetime.now().strftime("%H%M%S")
        latex_generator.generate_chart_image("db_scan_chart" + image_filename)
        plt.savefig(LATEX_RESULTS_DIR + "/db_scan_chart" + image_filename)
        plt.close()

    plt.show()


def _set_subplot(axs, row: int, column: int,
                 score: Tuple[List[float], List[float]], min_sample: float) -> None:
    axs[row, column].plot(score[0], score[1])
    axs[row, column].set_title("min sample: " + str(min_sample))
