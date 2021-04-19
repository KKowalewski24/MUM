import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np

from module.bayes import bayes_classification
from module.decision_tree import decision_tree_classification
from module.k_nearest_neighbors import knn_classification
from module.reader import read_data_set_from_csv_file
from module.support_vector_machine import svm_classification

"""
Sample usage:
    python main.py
    python main.py -s
"""

# VAR ------------------------------------------------------------------------ #

DATA_SET_HEART_FILENAME = "data/heart.csv"
DATA_SET_HEART_Y_COLUMN = "target"
DATA_SET_WEATHER_FILENAME = "data/weatherAUS.csv"
DATA_SET_WEATHER_Y_COLUMN = "RainTomorrow"
DATA_SET_COVID_FILENAME = "data/covid-19-symptoms-checker.csv"
DATA_SET_COVID_Y_COLUMN = ""


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save
    data_sets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
        read_data_set_from_csv_file(
            DATA_SET_HEART_FILENAME, DATA_SET_HEART_Y_COLUMN
        ),
        read_data_set_from_csv_file(
            DATA_SET_WEATHER_FILENAME, DATA_SET_WEATHER_Y_COLUMN
        )
    ]

    display_classifier_name("k-nearest neighbors classifier")
    knn_classification(data_sets, save_latex)

    display_classifier_name("naive Bayes classifier")
    bayes_classification(data_sets, save_latex)

    display_classifier_name("support vector machine classifier")
    decision_tree_classification(data_sets, save_latex)

    display_classifier_name("decision trees and random forests classifier")
    svm_classification(data_sets, save_latex)

    display_finish()


# DEF ------------------------------------------------------------------------ #
def display_classifier_name(name: str) -> None:
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
