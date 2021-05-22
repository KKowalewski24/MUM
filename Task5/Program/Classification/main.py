import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, \
    roc_curve
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier

from module.LatexGenerator import LatexGenerator
from module.reader import read_gestures_ds, read_heart_ds, read_weather_AUS

"""
Sample usage:
    python main.py
    python main.py -s
"""

# VAR ------------------------------------------------------------------------ #
LATEX_RESULTS_DIR = "latex_results"
latex_generator: LatexGenerator = LatexGenerator(LATEX_RESULTS_DIR)

classifiers_configuration = {
    "Heart": (read_heart_ds(), {
        "knn": KNeighborsClassifier(n_neighbors=9, p=2),
        "bayes": naive_bayes.GaussianNB(),
        "svm": svm.SVC(kernel="poly", C=1.6, gamma=0.0001),
        "random_forest": RandomForestClassifier(n_jobs=-1, min_samples_leaf=10, n_estimators=50,
                                                max_samples=0.5, random_state=47)
    }),
    "Gestures": (read_gestures_ds(), {
        "knn": KNeighborsClassifier(n_neighbors=9, p=2),
        "bayes": naive_bayes.GaussianNB(),
        "svm": svm.SVC(kernel="rbf", C=2.0, gamma=0.0001),
        "random_forest": RandomForestClassifier(n_jobs=-1, min_samples_leaf=8, n_estimators=500,
                                                random_state=47)
    }),
    "Weather": (read_weather_AUS(), {
        "knn": KNeighborsClassifier(n_neighbors=9, p=2),
        "bayes": naive_bayes.GaussianNB(),
        "svm": svm.SVC(kernel="rbf", C=1.6, gamma=0.001),
        "random_forest": RandomForestClassifier(n_jobs=-1, max_depth=5, n_estimators=200,
                                                max_samples=0.05, random_state=47)
    })
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    save_latex: bool = args.save

    for config in classifiers_configuration:
        display_header(config)
        data_set, classifiers = classifiers_configuration[config]
        metrics = {}
        for classifier in classifiers:
            print("\t", classifier)
            metrics[classifier] = evaluate_classifier(data_set, classifiers[classifier])
        if save_latex:
            save_metrics(metrics, config)

    display_finish()


# DEF ------------------------------------------------------------------------ #
def evaluate_classifier(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                        classifier) -> Dict:
    X_train, X_test, y_train, y_test = data_set
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    if type(classifier) is svm.SVC:
        y_proba = classifier.decision_function(X_test)
        if len(y_proba.shape) == 1:
            y_proba = np.stack([np.zeros((len(y_proba),)), y_proba], axis=1)
    else:
        y_proba = classifier.predict_proba(X_test)

    results = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "accuracy": np.round(accuracy_score(y_test, y_pred), 4),
        "recall": np.round(recall_score(y_test, y_pred, average=None), 4),
        "precision": np.round(precision_score(y_test, y_pred, average=None), 4),
        "roc_curves": [roc_curve(y_test, y_proba[:, i], pos_label=i) for i in np.unique(y_test)],
        "learning_curve": learning_curve(classifier, X_train, y_train, n_jobs=-1,
                                         train_sizes=np.linspace(0.1, 1.0, 10))
    }

    return results


def save_metrics(metrics, filename_prefix):
    # save tables with confusion matrices
    for classifier in metrics:
        matrix = metrics[classifier]["confusion_matrix"]
        latex_generator.generate_vertical_table(
            matrix[0], matrix[1:], filename_prefix + "_" + classifier + "_confusion_matrix"
        )

    # save tables with basic metrics
    if len(list(metrics.values())[0]["recall"]) == 2:
        matrix = [
            [classifier,
             metrics[classifier]["accuracy"],
             metrics[classifier]["recall"][1],
             metrics[classifier]["recall"][0],
             metrics[classifier]["precision"][1]]
            for classifier in metrics]
        latex_generator.generate_vertical_table(
            ["Classifier", "Accuracy", "Sensitivity", "Specificity", "Precision"],
            matrix, filename_prefix + "_basic_metrics"
        )
    else:
        matrix = [
            [classifier,
             metrics[classifier]["accuracy"],
             str(metrics[classifier]["recall"]),
             str(metrics[classifier]["precision"])]
            for classifier in metrics]
        latex_generator.generate_vertical_table(
            ["Classifier", "Accuracy", "Sensitivities", "Precisions"],
            matrix, filename_prefix + "_basic_metrics"
        )

    # save chart with ROC curve
    number_of_roc_curves = len(list(metrics.values())[0]["roc_curves"])
    if number_of_roc_curves == 2:
        for classifier in metrics:
            fpr, tpr, _ = metrics[classifier]["roc_curves"][1]
            plt.plot(fpr, tpr, label=classifier)
        plt.legend()
    else:
        for i in range(number_of_roc_curves):
            plt.subplot(int(np.ceil(number_of_roc_curves / 2)), 2, i + 1)
            plt.title("class: " + str(i))
            for classifier in metrics:
                fpr, tpr, _ = metrics[classifier]["roc_curves"][i]
                plt.plot(fpr, tpr, label=classifier)
            plt.legend()
    plt.show()

    # save charts with learning curves
    for classifier, i in zip(metrics, range(len(metrics))):
        plt.subplot(int(np.ceil(len(metrics) / 2)), 2, i + 1)
        plt.title(classifier)
        train_sizes_abs, train_scores, test_scores, = metrics[classifier]["learning_curve"]
        plt.plot(train_sizes_abs, np.average(train_scores, axis=1), label="train")
        plt.plot(train_sizes_abs, np.average(test_scores, axis=1), label="test")
        plt.legend()
    plt.show()


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
