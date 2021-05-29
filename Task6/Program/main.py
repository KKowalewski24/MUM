import subprocess
import sys
import numpy as np
from argparse import ArgumentParser, Namespace

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from module.reader import read_company_bankruptcy_prediction, read_student_alcohol_consumption, \
    read_wafer_manufacturing_anomalies

from module.correlation_based_feature_selection import correlation_based_feature_selection
from module.principal_component_analysis import principal_component_analysis
from module.variance_analysis import variance_analysis
from module.singular_value_decomposition import singular_value_decomposition

from sklearn.metrics import accuracy_score, recall_score, precision_score
"""
Sample usage:
    python main.py
    python main.py -s
"""

# VAR ------------------------------------------------------------------------ #
classifiers_per_datasets = {
    "student_alcohol_consumption": {
        "knn":
        KNeighborsClassifier(n_neighbors=3),
        "random_forest":
        RandomForestClassifier(n_jobs=-1,
                               max_depth=4,
                               n_estimators=50,
                               random_state=47)
    },
    "company_bankruptcy_prediction": {
        "knn":
        KNeighborsClassifier(n_neighbors=3),
        "random_forest":
        RandomForestClassifier(n_jobs=-1,
                               max_depth=4,
                               n_estimators=50,
                               random_state=47)
    },
    "wafer_manufacturing_anomalies": {
        "knn":
        KNeighborsClassifier(n_neighbors=3),
        "random_forest":
        RandomForestClassifier(n_jobs=-1,
                               max_depth=4,
                               n_estimators=50,
                               random_state=47)
    }
}

datasets_config = {
    "student_alcohol_consumption": read_student_alcohol_consumption(),
    "company_bankruptcy_prediction": read_company_bankruptcy_prediction(),
    "wafer_manufacturing_anomalies": read_wafer_manufacturing_anomalies()
}

dim_reduction_methods = [
    principal_component_analysis, singular_value_decomposition,
    variance_analysis, correlation_based_feature_selection
]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save

    for dim_reduction_method in dim_reduction_methods:
        print(dim_reduction_method.__name__)

        # prepare datasets - each method should add new variants (applying some
        # dimentionality reduction) after 'original' variant (no dimentionality
        # reduction)
        datasets = {
            ds_name: {
                "orignal": (datasets_config[ds_name][0].copy(),
                            datasets_config[ds_name][1].copy(),
                            datasets_config[ds_name][2].copy(),
                            datasets_config[ds_name][3].copy()),
            }
            for ds_name in datasets_config.keys()
        }
        dim_reduction_method(datasets, save_latex)

        # classification
        for ds_name in datasets:
            for variant_name in datasets[ds_name]:
                for classifier_name in classifiers_per_datasets[ds_name]:
                    print("\t", ds_name, "\t", variant_name, "\t",
                          classifier_name)
                    classifier = classifiers_per_datasets[ds_name][
                        classifier_name]
                    X_train, X_test, y_train, y_test = datasets[ds_name][
                        variant_name]
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    print("\t\taccuracy:", accuracy_score(y_test, y_pred))
                    print(
                        "\t\trecall:",
                        list(
                            np.round(
                                recall_score(y_test, y_pred, average=None),
                                3)))
                    print(
                        "\t\tprecision:",
                        list(
                            np.round(
                                precision_score(y_test,
                                                y_pred,
                                                average=None,
                                                zero_division=0), 3)))

    display_finish()


# DEF ------------------------------------------------------------------------ #
def display_header(name: str) -> None:
    print(
        "------------------------------------------------------------------------"
    )
    print(name)
    print()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-s",
        "--save",
        default=False,
        action="store_true",
        help="Create LaTeX source code based on generated data")

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
    print(
        "------------------------------------------------------------------------"
    )
    print("FINISHED")
    print(
        "------------------------------------------------------------------------"
    )


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()
