import subprocess
import sys
from argparse import ArgumentParser, Namespace

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from module.reader import read_company_bankruptcy_prediction, read_student_alcohol_consumption, \
    read_wafer_manufacturing_anomalies

"""
Sample usage:
    python main.py
    python main.py -s
"""

# VAR ------------------------------------------------------------------------ #
# TODO SET PROPER PARAMS FOR CLASSIFIERS
datasets_configuration = {
    "student_alcohol_consumption": (
        read_student_alcohol_consumption(), {
            "knn": KNeighborsClassifier(n_neighbors=9, p=2),
            "random_forest": RandomForestClassifier(
                n_jobs=-1, min_samples_leaf=10, n_estimators=50,
                max_samples=0.5, random_state=47
            )
        }),
    "company_bankruptcy_prediction": (
        read_company_bankruptcy_prediction(), {
            "knn": KNeighborsClassifier(n_neighbors=9, p=2),
            "random_forest": RandomForestClassifier(
                n_jobs=-1, min_samples_leaf=10, n_estimators=50,
                max_samples=0.5, random_state=47
            )
        }),
    "wafer_manufacturing_anomalies": (
        read_wafer_manufacturing_anomalies(), {
            "knn": KNeighborsClassifier(n_neighbors=9, p=2),
            "random_forest": RandomForestClassifier(
                n_jobs=-1, min_samples_leaf=10, n_estimators=50,
                max_samples=0.5, random_state=47
            )
        })

}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save

    for dataset_config in datasets_configuration:
        display_header(dataset_config + " data set")
        data_set, classifiers = datasets_configuration[dataset_config]
        for classifier in classifiers:
            print("\t", classifier)
            classifiers[classifier]

    display_finish()


# DEF ------------------------------------------------------------------------ #
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
