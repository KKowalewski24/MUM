from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from module.LatexGenerator import LatexGenerator

METRICS: Dict[int, str] = {1: "Manhattan", 2: "Euclidean"}
MIN_SAMPLES = np.arange(2, 10, 1)
EPSILONS = np.arange(0.1, 10, 0.1).round(2)

LATEX_RESULTS_DIR = "db_scan"
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def db_scan_clustering(data_set: np.ndarray, data_set_name: str, save_latex: bool = False) -> None:
    plt.suptitle("dataset: " + data_set_name)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for metric in reversed(METRICS):
        for j in range(len(MIN_SAMPLES)):
            plt.subplot(4, 2, j + 1, title="")
            silhouette_scores: Tuple[List[float], List[float]] = ([], [])

            for epsilon in EPSILONS:
                db_scan: DBSCAN = DBSCAN(
                    metric='minkowski', p=metric, min_samples=MIN_SAMPLES[j], eps=epsilon
                )
                cluster_labels = db_scan.fit_predict(data_set)

                is_all_items_same = np.all(cluster_labels == cluster_labels[0])
                if not is_all_items_same:
                    score = round(silhouette_score(data_set, cluster_labels), 4)
                    silhouette_scores[0].append(epsilon)
                    silhouette_scores[1].append(score)
                    print("Epsilon:\t", epsilon, "\t", "Silhouette:\t", score)

            plt.plot(
                silhouette_scores[0], silhouette_scores[1],
                label="Metric: " + METRICS[metric]
            )
            plt.legend()

    plt.show()
