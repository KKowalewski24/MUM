import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from module.LatexGenerator import LatexGenerator
from datetime import datetime

LATEX_RESULTS_DIR = "em_algo"
n_clusters_range = [4, 6, 9, 15, 20]
n_iters_range = [40, 60, 80, int(1e2), 200, int(1e3), int(1e4), int(1e5), int(1e6), int(1e7), int(1e8), int(1e9), int(1e10)]
covariance_types = {0: "full", 1: "tied", 2: "diag", 3: "spherical"}
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def expectation_maximization_clustering(data_set: np.ndarray, data_set_name: str,
                                        save_latex: bool = False) -> None:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("dataset: " + data_set_name)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for n_clusters in n_clusters_range:
        latex_data = []
        for max_it in n_iters_range:
            latex_data.append([max_it])

        for variant_id in covariance_types:
            plt.subplot(2, 2, variant_id + 1, title="covariance: " + covariance_types[variant_id])
            plt.grid()
            plt.xlabel('Max iterations')
            plt.ylabel('Silhouette coefficient value')
            score = []
            latex_row_index = 0
            for max_iter in n_iters_range:
                y = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    max_iter=max_iter).fit_predict(data_set)
                silhouette = silhouette_score(data_set, y)
                latex_data[latex_row_index].append(round(silhouette, 3))
                score.append(silhouette)
                latex_row_index += 1
            plt.plot(n_iters_range, score, label=str(n_clusters))
        if save_latex:
            file_name = data_set_name+"_"+str(n_clusters)+"_"+datetime.now().strftime("%H%M%S")
            latex_generator.generate_vertical_table(header_names=["Max iterations", "full", "tied", "diag", "spherical"]
                                                    , body_values=latex_data, filename=file_name)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, title="\n Clusters: ", bbox_to_anchor=(1, 1), loc="upper right", ncol=1)

    if save_latex:
        base_filename = "_" + data_set_name
        image_filename = base_filename + "-" + datetime.now().strftime("%H%M%S")
        latex_generator.generate_chart_image("em_scan_chart" + image_filename)
        plt.savefig(LATEX_RESULTS_DIR + "/em_scan_chart" + image_filename)
    if not save_latex:
        plt.show()

    plt.close()



