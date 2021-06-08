import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

variants = {
    'single': ['euclidean', 'manhattan'],
    'complete': ['euclidean', 'manhattan'],
    'average': ['euclidean', 'manhattan'],
    'ward': ['euclidean']
}
n_clusters_range = np.arange(15, 1, -1)


def agglomerative_clustering(data_set: np.ndarray,
                             data_set_name: str,
                             save_latex: bool = False) -> None:
    plt.suptitle("dataset: " + data_set_name)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for variant_id, linkage in zip(range(len(variants)), variants.keys()):
        plt.subplot(2, 2, variant_id + 1, title="linkage: " + linkage)
        plt.grid()
        plt.xlabel('N clusters')
        plt.ylabel('Silhouette score')
        for affinity in variants[linkage]:
            score = []
            for n_clusters in n_clusters_range:
                y = AgglomerativeClustering(
                    n_clusters=n_clusters, affinity=affinity,
                    linkage=linkage).fit_predict(data_set)
                score.append(silhouette_score(data_set, y))
            plt.plot(n_clusters_range,
                     score,
                     label=affinity + ' max=' + str(np.round(np.max(score), 2)))
            plt.legend()
    plt.show()
