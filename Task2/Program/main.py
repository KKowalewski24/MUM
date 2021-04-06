import subprocess
import sys

from module.imputation import hot_deck, interpolate, mean, regression
from module.reader import read
from module.statistics import calculate_statistics

"""
"""


def main() -> None:
    for ds, label in zip(read([5, 15, 30, 45]), ['5%', '15%', '30%', '45%']):
        print(label, "missing values\n\n")

        display_separator()
        print("List wise deletion")
        calculate_statistics(ds.dropna())

        display_separator()
        print("Mean imputation")
        calculate_statistics(mean(ds))

        display_separator()
        print("Interpolation")
        calculate_statistics(interpolate(ds))

        display_separator()
        print("Hot deck")
        calculate_statistics(hot_deck(ds))

        display_separator()
        print("Regression")
        # calculate_statistics(regression(ds))


def display_separator() -> None:
    print("------------------------------------------------------------------------")


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
