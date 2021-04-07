import subprocess
import sys
from argparse import ArgumentParser, Namespace

from module.imputation import hot_deck, interpolate, mean, regression
from module.reader import read
from module.statistics import calculate_statistics

"""
"""


def main() -> None:
    args = prepare_args()
    save_to_files: bool = args.save

    for ds, label in zip(read([5, 15, 30, 45]), ['5%', '15%', '30%', '45%']):
        print(label, "missing values\n\n")
        calculate_statistics(ds.dropna(), save_to_files, label, "List wise deletion")
        calculate_statistics(mean(ds), save_to_files, label, "Mean imputation")
        calculate_statistics(interpolate(ds), save_to_files, label, "Interpolation")
        calculate_statistics(hot_deck(ds), save_to_files, label, "Hot deck")
        # calculate_statistics(regression(ds), save_to_files, label, "Regression")


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
