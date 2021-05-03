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
    scores: List[List[float]] = []
    for cluster_value in CLUSTERS_NUMBER:
        scores.append([])
        k_means = KMeans(
            n_clusters=cluster_value
        )
        cluster_labels = k_means.fit_predict(data_set)
        is_all_items_same = np.all(cluster_labels == cluster_labels[0])
        if not is_all_items_same:
            score = round(silhouette_score(data_set, cluster_labels), 4)
            scores.append(score)
            print("Clusters number: " + str(cluster_value) + "\tSilhouette:\t" + str(score))

   