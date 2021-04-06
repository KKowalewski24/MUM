import subprocess
import sys
from module.reader import read
from module.imputation import mean, interpolate, hot_deck, regression
from module.statistics import calculate_statistics

"""
"""


# VAR ------------------------------------------------------------------------ #

# MAIN ----------------------------------------------------------------------- #
def main() -> None:

    for ds, label in zip(read([5, 15, 30, 45]), ['5%', '15%', '30%', '45%']):
        print("------------------------------------------------------------------------")
        print(label, "missing values\n\n")

        print("List wise deletion")
        calculate_statistics(ds.dropna())

        print("Mean imputation")
        calculate_statistics(mean(ds))

        print("Interpolation")
        calculate_statistics(interpolate(ds))

        print("Hot deck")
        calculate_statistics(hot_deck(ds))

        print("Regression")
        # calculate_statistics(regression(ds))
        print("------------------------------------------------------------------------")

    display_finish()


# DEF ------------------------------------------------------------------------ #

# UTIL ----------------------------------------------------------------------- #
def check_types_check_style() -> None:
    subprocess.call(["mypy", "."])
    subprocess.call(["flake8", "."])


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
    else:
        main()
