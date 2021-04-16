import subprocess
import sys
from typing import Dict

import pandas as pd

from module.reader import read_csv_data_sets

"""
"""


# VAR ------------------------------------------------------------------------ #

# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    # TODO
    data_sets: Dict[int, pd.DataFrame] = read_csv_data_sets([""])

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
