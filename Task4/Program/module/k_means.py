from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "k_means"
CLUSTERS_NUMBER = range(2, 31)
CHARTS_NUMBER = 4

latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def k_means_clustering(data_set: np.ndarray, data_set_name: str,
                       save_latex: bool = False) -> None:
    scores: List[Tuple[List[float], List[float]]] = []
    for cluster_value in CLUSTERS_NUMBER:
        # scores.append(([], []))
        k_means = KMeans(
            n_clusters=cluster_value
        )
        cluster_labels = k_means.fit_predict(data_set)
        is_all_items_same = np.all(cluster_labels == cluster_labels[0])
        if not is_all_items_same:
            score = round(silhouette_score(data_set, cluster_labels), 4)
            scores.append((cluster_value, score))
            print("Clusters number: " + str(cluster_value) + "\tSilhouette:\t" + str(score))

    if save_latex :
        _save_score(scores, data_set_name)   
        _draw_and_save_chart(scores, data_set_name, 0)


def _save_score(score: List[List[float]], data_set_name: str) -> None:
    filename_description = "_" + data_set_name
    latex_generator.generate_horizontal_table(
        ["Silhouette"], score,
        "kmeans_table" + filename_description
    )

def _draw_and_save_chart(score: List[List[float]], data_set_name: str, order_number: int) -> None:
    base_filename = "_" + data_set_name
    image_filename = base_filename + str(order_number) + "-" + datetime.now().strftime("%H%M%S")
    
    plt.plot(*zip(*score), "red")
    latex_generator.generate_chart_image("kmeans_chart" + image_filename)
    plt.savefig(LATEX_RESULTS_DIR + "/kmeans_chart" + image_filename)
    plt.close()

    plt.show()
