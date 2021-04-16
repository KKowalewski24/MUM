import subprocess
import sys
from typing import Dict

import pandas as pd

from module.bayes import bayes_classification
from module.decision_tree import decision_tree_classification
from module.k_nearest_neighbors import knn_classification
from module.reader import read_csv_data_sets
from module.support_vector_machine import svm_classification

"""
"""

# VAR ------------------------------------------------------------------------ #
# TODO
DATA_SET_FILENAMES = ["", "", ""]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    data_sets: Dict[int, pd.DataFrame] = read_csv_data_sets(DATA_SET_FILENAMES)

    display_classifier_name("k-nearest neighbors")
    knn_classification(data_sets)

    display_classifier_name("naive Bayes classifier")
    bayes_classification(data_sets)

    display_classifier_name("support vector machine")
    decision_tree_classification(data_sets)

    display_classifier_name("decision trees and random forests")
    svm_classification(data_sets)

    display_finish()


# DEF ------------------------------------------------------------------------ #
def display_classifier_name(name: str) -> None:
    print("------------------------------------------------------------------------")
    print(name)


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
