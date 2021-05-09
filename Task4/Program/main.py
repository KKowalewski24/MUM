import subprocess
import sys
from argparse import ArgumentParser, Namespace

import numpy as np

from module.agglomerative import agglomerative_clustering
from module.db_scan import db_scan_clustering
from module.expectation_maximization import expectation_maximization_clustering
from module.k_means import k_means_clustering
from module.reader import present_data_sets, read_iris_ds, read_mall_customers, read_moons_ds

"""
Sample usage:
    python main.py
    python main.py -s
"""


# VAR ------------------------------------------------------------------------ #

# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save
    if args.present:
        present_data_sets()

    process_clustering(read_iris_ds(), "Iris", save_latex)
    # process_clustering(read_mall_customers(), "Customers", save_latex)
    # process_clustering(read_moons_ds(), "Moons", save_latex)
    display_finish()


# DEF ------------------------------------------------------------------------ #
def process_clustering(data_set: np.ndarray, data_set_name: str,
                       save_latex: bool) -> None:
    # display_header(data_set_name + " data set")
    # display_header("K Means")
    # k_means_clustering(data_set, data_set_name, save_latex)
    # display_header("Agglomerative")
    # agglomerative_clustering(data_set, data_set_name, save_latex)
    display_header("Expectation Maximization")
    expectation_maximization_clustering(data_set, data_set_name, save_latex)
    # display_header("DBSCAN")
    # db_scan_clustering(data_set, data_set_name, True, save_latex)
    # db_scan_clustering(data_set, data_set_name, False, save_latex)


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
