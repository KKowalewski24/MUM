import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

from module.LatexGenerator import LatexGenerator
from module.CFS.correlation_based_feature_selection import correlation_based_feature_selection
from module.principal_component_analysis import principal_component_analysis
from module.reader import read_company_bankruptcy_prediction, read_student_alcohol_consumption, \
    read_wafer_manufacturing_anomalies
from module.singular_value_decomposition import singular_value_decomposition
from module.variance_analysis import variance_analysis

"""
Sample usage:
    python main.py
    python main.py -s
"""

# VAR ------------------------------------------------------------------------ #
classifiers_per_datasets = {
    "student_alcohol_consumption": {
        "knn": KNeighborsClassifier(n_neighbors=3),
        "random_forest": RandomForestClassifier(n_jobs=-1, max_depth=4,
                                                n_estimators=50, random_state=47)
    },
    "company_bankruptcy_prediction": {
        "knn": KNeighborsClassifier(n_neighbors=3),
        "random_forest": RandomForestClassifier(n_jobs=-1, max_depth=4,
                                                n_estimators=50, random_state=47)
    },
    "wafer_manufacturing_anomalies": {
        "knn": KNeighborsClassifier(n_neighbors=3),
        "random_forest": RandomForestClassifier(n_jobs=-1, max_depth=4,
                                                n_estimators=50, random_state=47)
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
    latex_generator: LatexGenerator = LatexGenerator("results")

    for dim_reduction_method in dim_reduction_methods:
        display_header(dim_reduction_method.__name__)

        # prepare datasets - each method should add new variants (applying some dimensionality
        # reduction) after 'original' variant (no dimensionality reduction)
        datasets = {
            ds_name: {
                "original": (datasets_config[ds_name][0].copy(),
                             datasets_config[ds_name][1].copy(),
                             datasets_config[ds_name][2].copy(),
                             datasets_config[ds_name][3].copy()),
            }
            for ds_name in datasets_config.keys()
        }
        dim_reduction_method(datasets, save_latex)

        # classification
        for ds_name in datasets:
            params_accuracy_values: List[List[Union[str, float]]] = []
            for variant_name in datasets[ds_name]:
                accuracy_values: List[Union[str, float]] = []
                for classifier_name in classifiers_per_datasets[ds_name]:
                    print("\t", ds_name + ",", "\t", variant_name + ",", "\t", classifier_name)

                    X_train, X_test, y_train, y_test = datasets[ds_name][variant_name]
                    classifier = classifiers_per_datasets[ds_name][classifier_name]
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)

                    accuracy = round(accuracy_score(y_test, y_pred), 3)
                    accuracy_values.append(classifier_name)
                    accuracy_values.append(accuracy)
                    print("\t\taccuracy:", accuracy)
                    print(
                        "\t\trecall:",
                        list(np.round(recall_score(y_test, y_pred, average=None), 3))
                    )
                    print(
                        "\t\tprecision:",
                        list(np.round(precision_score(
                            y_test, y_pred, average=None, zero_division=0), 3)
                        ))
                    print()

                params_accuracy_values.append([variant_name] + accuracy_values)

            if save_latex:
                latex_generator.generate_vertical_table(
                    ["Parametry metody", "Klasyfikator", "Accuracy", "Klasyfikator", "Accuracy"],
                    params_accuracy_values, "table_" + dim_reduction_method.__name__ + "_" + ds_name
                )

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
