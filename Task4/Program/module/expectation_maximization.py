import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from module.LatexGenerator import LatexGenerator
from datetime import datetime

LATEX_RESULTS_DIR = "em_algo"
n_clusters_range = [4, 6, 9, 15, 20]
n_iters_range = [int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)]
covariance_types = {0: "full", 1: "tied", 2: "diag", 3: "spherical"}
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def expectation_maximization_clustering(data_set: np.ndarray, data_set_name: str,
                                        save_latex: bool = False) -> None:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("dataset: " + data_set_name)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    print("Dataset:"+data_set_name)
    for variant_id in covariance_types:
        print(covariance_types[variant_id]+":")
        plt.subplot(2, 2, variant_id + 1, title="covariance: " + covariance_types[variant_id])
        plt.grid()
        plt.xlabel('Max iterations')
        plt.ylabel('Silhouette coefficient value')
        for n_clusters in n_clusters_range:
            print("Number of clusters: "+str(n_clusters))
            score = []
            for max_iter in n_iters_range:
                print("Number of max iterations: "+str(max_iter))
                y = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    max_iter=max_iter).fit_predict(data_set)
                silhouette = silhouette_score(data_set, y)
                print("Score: " + str(silhouette))
                score.append(silhouette)
            plt.plot(n_iters_range, score, label=str(n_clusters))
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, title="Clusters: ", bbox_to_anchor=(1,1), loc="upper right",
                ncol=1)
    plt.show()
    if save_latex:
        base_filename = "_" + data_set_name
        image_filename = base_filename + "-" + datetime.now().strftime("%H%M%S")
        latex_generator.generate_chart_image("db_scan_chart" + image_filename)
        plt.savefig(LATEX_RESULTS_DIR + "/db_scan_chart" + image_filename)
        plt.close()



