import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def db_scan_clustering(data_set: pd.DataFrame, data_set_name: str,
                       is_euclidean_metric: bool, save_latex: bool = False) -> None:
    db_scan: DBSCAN = DBSCAN(eps=0.5, p=2 if is_euclidean_metric else 1)
    # y_db_scan = db_scan.fit_predict(data_set)
    # print("Silhouette:\t" + str(silhouette_score(data_set, y_db_scan)))
