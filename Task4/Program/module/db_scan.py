import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

MIN_SAMPLES_RANGE = np.arange(2, 10, 1)
EPSILON_RANGE = np.arange(0.1, 10, 0.1)


def db_scan_clustering(data_set: np.ndarray, data_set_name: str,
                       is_euclidean_metric: bool, save_latex: bool = False) -> None:
    for min_sample in MIN_SAMPLES_RANGE:
        for epsilon in EPSILON_RANGE:
            db_scan: DBSCAN = DBSCAN(
                min_samples=min_sample, eps=epsilon, p=2 if is_euclidean_metric else 1
            )
            cluster_labels = db_scan.fit_predict(data_set)

            is_all_items_same = np.all(cluster_labels == cluster_labels[0])
            if not is_all_items_same:
                score = round(silhouette_score(data_set, cluster_labels), 4)
                print("Silhouette:\t" + str(score))
