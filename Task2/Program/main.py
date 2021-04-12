import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace

from module.imputation import hot_deck, interpolate, mean, regression
from module.latex_generator import RESULTS_DIR_NAME
from module.reader import read
from module.statistics import calculate_statistics


def main() -> None:
    args = prepare_args()
    save_to_files: bool = args.save
    create_dir_for_results(save_to_files)
    categorical_columns = [1, 2, 5, 6, 8, 10, 11, 12, 13]

    for ds, label in zip(read([5, 15, 30, 45]), ['5%', '15%', '30%', '45%']):
        print("\n" + label, "missing values")
        calculate_statistics(ds.dropna(), save_to_files, label, "List wise deletion")
        calculate_statistics(mean(ds, categorical_columns), save_to_files, label, "Mean imputation")
        calculate_statistics(interpolate(ds, categorical_columns), save_to_files, label, "Interpolation")
        calculate_statistics(hot_deck(ds), save_to_files, label, "Hot deck")
        calculate_statistics(regression(ds, categorical_columns), save_to_files, label, "Regression")


def create_dir_for_results(save_to_files: bool) -> None:
    if not save_to_files:
        return

    if not os.path.exists(RESULTS_DIR_NAME):
        os.makedirs(RESULTS_DIR_NAME)


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true",
        help="Convert printed data to LateX tables and save them to files"
    )

    return arg_parser.parse_args()


def check_types_check_style() -> None:
    subprocess.call(["mypy", "."])
    subprocess.call(["flake8", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    else:
        main()
