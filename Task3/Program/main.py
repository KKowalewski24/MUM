import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np

from module.bayes import bayes_classification
from module.decision_tree import decision_tree_classification
from module.k_nearest_neighbors import knn_classification
from module.reader import read_gestures_ds, read_heart_ds, read_weather_AUS
from module.support_vector_machine import svm_classification

"""
Sample usage:
    python main.py
    python main.py -s
"""


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save
    display_header("Heart data set")
    process_classifiers(read_heart_ds(), "heart", save_latex)
    display_header("Gestures data set")
    process_classifiers(read_gestures_ds(), "gestures", save_latex)
    display_header("Weather data set")
    process_classifiers(read_weather_AUS(), "weather", save_latex)
    display_finish()


# DEF ------------------------------------------------------------------------ #
def process_classifiers(data_sets: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                        data_set_name: str, save_latex: bool) -> None:
    # display_header("k-nearest neighbors classifier")
    # knn_classification(data_sets, True, data_set_name, save_latex)
    # knn_classification(data_sets, False, data_set_name, save_latex)

    # display_header("naive Bayes classifier")
    # bayes_classification(data_sets, save_latex)

    display_header("support vector machine classifier")
    svm_classification(data_sets, save_latex)

    # display_header("decision trees and random forests classifier")
    # decision_tree_classification(data_sets, save_latex)



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
