import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple, Dict

import numpy as np

from module.LatexGenerator import LatexGenerator
from module.reader import read_mall_customers, read_iris_ds, read_moons_ds
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, fowlkes_mallows_score

"""
Sample usage:
    python main.py
    python main.py -s
"""


# VAR ------------------------------------------------------------------------ #
LATEX_RESULTS_DIR = "latex_results"
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)

clusters_configuration = {
    "Iris": (read_iris_ds(), {
        "k_means": KMeans(n_clusters=3),
        # "agglomerative": AgglomerativeClustering(n_clusters=1, affinity=1, linkage=1),
        "agglomerative":  KMeans(n_clusters=3),
        "expectation_maximization": GaussianMixture(n_components=4, covariance_type="full", max_iter=200),
        "db_scan":  DBSCAN(min_samples=7, eps=0.9, metric='minkowski', p=2)
    }),
    "Customers": (read_mall_customers(), {
        "k_means": KMeans(n_clusters=6),
        "agglomerative": KMeans(n_clusters=6),
        # "agglomerative": AgglomerativeClustering(n_clusters=1, affinity=1, linkage=1),
        "expectation_maximization": GaussianMixture(n_components=6, covariance_type="diag", max_iter=200),
        "db_scan":  DBSCAN(min_samples=7, eps=23, metric='minkowski', p=2)
    }),
    "Moons": (read_moons_ds(), {
        "k_means": KMeans(n_clusters=8),
        "agglomerative": KMeans(n_clusters=8),
        # "agglomerative": AgglomerativeClustering(n_clusters=1, affinity=1, linkage=1),
        "expectation_maximization": GaussianMixture(n_components=9, covariance_type="full", max_iter=200),
        "db_scan":  DBSCAN(min_samples=5, eps=0.2, metric='minkowski', p=2)
    })
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save
    for config in clusters_configuration:
        display_header(config)
        data_set = clusters_configuration[config][0]
        classifiers = clusters_configuration[config][1]
        metrics = {}
        for classifier in classifiers:
            print("\t", classifier)
            metrics[classifier] = evaluate_classifier(data_set, config, classifiers[classifier])
        if save_latex:
            save_metrics(metrics, config)

    display_finish()


# DEF ------------------------------------------------------------------------ #
def evaluate_classifier(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], data_set_name,
                        classifier) -> Dict:
    data_set_labels = []
    if data_set_name == "Iris":
        data_set_labels = data_set[:, 4]
        data_set = data_set[:, 0:4]
    elif data_set_name == "Moons":
        data_set_labels = data_set[1]
        data_set = data_set[0]
    cluster_labels = classifier.fit_predict(data_set)

    if data_set_name != "Customers":
        results = {
            "silhouette": silhouette_score(data_set, cluster_labels),
            "calinski_harabasz": calinski_harabasz_score(data_set, cluster_labels),
            "davies_bouldin": davies_bouldin_score(data_set, cluster_labels),
            "rand_score": rand_score(data_set_labels, cluster_labels),
            "fowlkes_mallows": fowlkes_mallows_score(data_set_labels, cluster_labels)
        }
    else:
        results = {
            "silhouette": silhouette_score(data_set, cluster_labels),
            "calinski_harabasz": calinski_harabasz_score(data_set, cluster_labels),
            "davies_bouldin": davies_bouldin_score(data_set, cluster_labels)
        }
    return results


def save_metrics(metrics, filename_prefix):
    # save tables with basic metrics
    if filename_prefix == "Customers":
        matrix = [
            [classifier,
             metrics[classifier]["silhouette"],
             metrics[classifier]["calinski_harabasz"],
             metrics[classifier]["davies_bouldin"]]
            for classifier in metrics]
        latex_generator.generate_vertical_table(
            ["Classifier", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"],
            matrix, filename_prefix + "_basic_metrics"
        )
    else:
        matrix = [
            [classifier,
             metrics[classifier]["silhouette"],
             metrics[classifier]["calinski_harabasz"],
             metrics[classifier]["davies_bouldin"],
             metrics[classifier]["rand_score"],
             metrics[classifier]["fowlkes_mallows"]]
            for classifier in metrics]
        latex_generator.generate_vertical_table(
            ["Classifier", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin", "Rand_score", "Fowlkes_Mallows"],
            matrix, filename_prefix + "_basic_metrics"
        )


def display_header(name: str) -> None:
    print("------------------------------------------------------------------------")
    print(name)
    print()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true",
        help="Create LaTeX source code based on generated data"
    )
    arg_parser.add_argument(
        "-pd", "--present", default=False, action="store_true",
        help="Present characteristic of data sets"
    )

    return arg_parser.parse_args()


# UTIL ----------------------------------------------------------------------- #
def check_types_check_style() -> None:
    subprocess.call(["mypy", "."])
    subprocess.call(["flake8", "."])


def compile_to_pyc() -> None:
    subprocess.call(["python", "-m", "compileall", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()
