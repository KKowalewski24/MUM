from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "k_means"
CLUSTERS_NUMBER = range(2, 31)
CLUSTERS_NUMBER_TO_MAX_ITER = [2, 6, 8, 19, 25] #te wartosci zostaly wybrane "na oko", bazujac na wykresach generowanych przy badaniu Cluster-number
MAX_ITER = [int(float("1e" + str(x))) for x in range(2, 12)] #2,12

latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)


def k_means_clustering(data_set: np.ndarray, data_set_name: str,
                       save_latex: bool = False) -> None:
    scores_clusters_numbers: List[Tuple[List[float], List[float]]] = []
    scores_iter_numbers: List[Tuple[List[float], List[float]]] = []

    for cluster_value in CLUSTERS_NUMBER:
        k_means = KMeans(
            n_clusters = cluster_value
        )
        cluster_labels = k_means.fit_predict(data_set)
        is_all_items_same = np.all(cluster_labels == cluster_labels[0])
        if not is_all_items_same:
            score = round(silhouette_score(data_set, cluster_labels), 4)
            scores_clusters_numbers.append((cluster_value, score))
            print("Clusters number: " + str(cluster_value) + "\tSilhouette:\t" + str(score))

    for cluster_value in CLUSTERS_NUMBER_TO_MAX_ITER:
        for iter_value in MAX_ITER:
            k_means = KMeans(
                n_clusters = cluster_value,
                max_iter = iter_value
            )
            cluster_labels = k_means.fit_predict(data_set)
            is_all_items_same = np.all(cluster_labels == cluster_labels[0])
            if not is_all_items_same:
                score = round(silhouette_score(data_set, cluster_labels), 4)
                # scores_iter_numbers.append((cluster_value, iter_value, score))
                scores_iter_numbers.append((iter_value, score))
                print("Clusters number: " + str(cluster_value) + "\tMax iterations number:\t" + str(iter_value)
                + "\tSilhouette:\t" + str(score))

    # if save_latex :
    #     _save_score(scores_clusters_numbers, data_set_name)   
    #     _draw_and_save_chart_clusters(scores_clusters_numbers, data_set_name, 0)
    if save_latex :
        # _save_score(scores_clusters_numbers, data_set_name)   
        _draw_and_save_chart(scores_clusters_numbers, scores_iter_numbers, data_set_name)


def _save_score(score: List[List[float]], data_set_name: str) -> None:
    filename_description = "_" + data_set_name
    latex_generator.generate_horizontal_table(
        ["Silhouette"], score,
        "kmeans_table" + filename_description
    )

def _draw_and_save_chart(score_clusters: List[List[float]], score_iters: List[List[float]], data_set_name: str) -> None:
    base_filename = "_" + data_set_name
    image_filename = base_filename + "_" + datetime.now().strftime("%H%M%S")
    colors = {2: "red", 6: "blue", 8: "green", 19: "purple", 25: "orange"}

#CLUSTER CHARTS
    plt.subplot(211)
    plt.subplots_adjust(left=0.1,
                        bottom=0.2,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.5)
    plt.plot(*zip(*score_clusters), "green")
    plt.ylabel("Silhouette")
    plt.xlabel("Clusters number")

#ITERATIONS CHARTS
    plt.subplot(212)
    index = 0
    for cluster_number in CLUSTERS_NUMBER_TO_MAX_ITER:
        plt.plot(*zip(*score_iters[index:index+10]), colors[cluster_number], label= str("Clusters: " + str(cluster_number)))
        index += 10

    plt.xscale('log')
    plt.ylabel("Silhouette")
    plt.xlabel("Max Iterations number")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper left', fontsize='xx-small', ncol=5, mode="expand")
    
    latex_generator.generate_chart_image("kmeans_chart" + image_filename)
    plt.savefig(LATEX_RESULTS_DIR + "/kmeans_chart" + image_filename)
    plt.close()
    plt.show()