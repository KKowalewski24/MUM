import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np
from module.reader import read_gestures_ds, read_heart_ds, read_weather_AUS
from sklearn import metrics, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

"""
Sample usage:
    python main.py
    python main.py -s
"""

# TODO SET PROPER PARAMETERS
classifiers_configuration = {
    "Heart": (read_heart_ds(), {
        "knn": KNeighborsClassifier(n_neighbors=9, p=2),
        "bayes": naive_bayes.GaussianNB(),
        # "svm": svm.SVC(kernel=kernel_function, C=c, gamma=gamma),
        # "decision_tree": DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
        #                                         max_depth=max_depth, random_state=47),
        # "random_forest": RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
        #                                         **best_params, random_state=47)
    }),
    "Gestures": (read_gestures_ds(), {
        "knn": KNeighborsClassifier(n_neighbors=9, p=2),
        "bayes": naive_bayes.GaussianNB(),
        # "svm": svm.SVC(kernel=kernel_function, C=c, gamma=gamma),
        # "decision_tree": DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
        #                                         max_depth=max_depth, random_state=47),
        # "random_forest": RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
        #                                         **best_params, random_state=47)
    }),
    "Weather": (read_weather_AUS(), {
        "knn": KNeighborsClassifier(n_neighbors=9, p=2),
        "bayes": naive_bayes.GaussianNB(),
        # "svm": svm.SVC(kernel=kernel_function, C=c, gamma=gamma),
        # "decision_tree": DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
        #                                         max_depth=max_depth, random_state=47),
        # "random_forest": RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
        #                                         **best_params, random_state=47)
    })
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save
    for config in classifiers_configuration:
        display_header(config)
        data_set = classifiers_configuration[config][0]
        classifiers = classifiers_configuration[config][1]
        for classifier in classifiers:
            print(classifier)
            calculate_metrics(data_set, classifiers[classifier])
            print()

    display_finish()


# DEF ------------------------------------------------------------------------ #
def calculate_metrics(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                      classifier) -> None:
    X_train, X_test, y_train, y_test = data_set
    classifier.fit(X_train, y_train)
    y_prediction = classifier.predict(X_test)
    accuracy = round(metrics.accuracy_score(y_test, y_prediction), 4)
    print(accuracy)


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
