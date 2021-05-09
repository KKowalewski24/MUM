import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
n_clusters_range = [4, 6, 9, 15, 20]
n_iters_range = [100, 1000, 10000, 100000]
covariance_types = {0: "full", 1: "tied", 2: "diag", 3: "spherical"}


def expectation_maximization_clustering(data_set: np.ndarray, data_set_name: str,
                                        save_latex: bool = False) -> None:
    fig = plt.figure()
    fig.suptitle("dataset: " + data_set_name)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for variant_id in covariance_types:
        plt.subplot(2, 2, variant_id + 1, title="covariance: " + covariance_types[variant_id])
        plt.grid()
        plt.xlabel('Max iterations')
        plt.ylabel('Silhouette coefficient value')
        for n_clusters in n_clusters_range:
            score = []
            for max_iter in n_iters_range:
                y = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    max_iter=max_iter).fit_predict(data_set)
                score.append(silhouette_score(data_set, y))
            plt.plot(n_iters_range, score, label=str(n_clusters))
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, title="Clusters: ", bbox_to_anchor=(1,1), loc="upper right",
                ncol=1)
    plt.show()



