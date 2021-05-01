from datetime import datetime
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
    for min_sample in MIN_SAMPLES_RANGE:
        # Tuple[List[epsilon], List[silhouette_score]]
        silhouette_scores: Tuple[List[float], List[float]] = ([], [])
        for epsilon in EPSILON_RANGE:
            db_scan: DBSCAN = DBSCAN(
                min_samples=min_sample, eps=epsilon, p=2 if is_euclidean_metric else 1
            )
            cluster_labels = db_scan.fit_predict(data_set)

            is_all_items_same = np.all(cluster_labels == cluster_labels[0])
            if not is_all_items_same:
                score = round(silhouette_score(data_set, cluster_labels), 4)
                silhouette_scores[0].append(epsilon)
                silhouette_scores[1].append(score)
                print("Silhouette:\t" + str(score))

        plt.plot(silhouette_scores[0], silhouette_scores[1])
        plt.title(
            data_set_name + (", Euclidean" if is_euclidean_metric else ", Manhattan")
            + " metric" + ", min sample: " + str(min_sample)
        )
        plt.xlabel("Epsilon")
        plt.ylabel("Silhouette Score")

        if save_latex:
            filename_description = "_" + data_set_name \
                                   + ("_eucl" if is_euclidean_metric else "_manh")
            image_filename = filename_description + "-" + datetime.now().strftime("%H%M%S")

            plt.close()

        plt.show()
