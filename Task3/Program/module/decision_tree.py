from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from module.LatexGenerator import LatexGenerator

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

latex_generator: LatexGenerator = LatexGenerator("decision_tree")


def _plot_accuracy(subplot_idx, train_acc_history, test_acc_history,
                   param_name, param_range):
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(subplot_idx)
    plt.ylabel("accuracy")
    plt.xlabel(param_name)
    x = range(len(param_range))
    plt.xticks(x, param_range)
    plt.plot(x, train_acc_history, label="train accuracy")
    plt.plot(x, test_acc_history, label="test accuracy")
    plt.annotate(
        str(round(np.max(test_acc_history), 3)),
        (np.argmax(test_acc_history) + 0.05, np.max(test_acc_history)))
    plt.plot([np.argmax(test_acc_history)], [np.max(test_acc_history)], 'go')
    plt.grid()
    plt.legend()


def decision_tree_classification(data_set: Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray],
                                 save_latex: bool) -> None:

    X_train, X_test, y_train, y_test = data_set

    min_samples_leaf_range = [
        2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000
    ]
    max_depth_range = [30, 25, 20, 18, 15, 13, 10, 8, 5, 3, 2, 1]
    n_estimators_range = [10, 20, 50, 80, 100, 200, 500]
    max_samples_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99]

    # find best decision tree regularization parameters

    train_acc_history = []
    test_acc_history = []
    for min_samples_leaf in min_samples_leaf_range:
        tree = DecisionTreeClassifier(random_state=47,
                                      min_samples_leaf=min_samples_leaf)
        tree.fit(X_train, y_train)
        train_acc_history.append(tree.score(X_train, y_train))
        test_acc_history.append(tree.score(X_test, y_test))
        print("min_samples_leaf:", min_samples_leaf, "\ttrain_acc:",
              train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
    _plot_accuracy(411, train_acc_history, test_acc_history,
                   "min_samples_leaf", min_samples_leaf_range)
    best_acc = np.max(test_acc_history)
    best_params = {
        "min_samples_leaf": min_samples_leaf_range[np.argmax(test_acc_history)]
    }

    train_acc_history = []
    test_acc_history = []
    for max_depth in max_depth_range:
        tree = DecisionTreeClassifier(random_state=47, max_depth=max_depth)
        tree.fit(X_train, y_train)
        train_acc_history.append(tree.score(X_train, y_train))
        test_acc_history.append(tree.score(X_test, y_test))
        print("max_depth:", max_depth, "\ttrain_acc:", train_acc_history[-1],
              "\ttest_acc:", test_acc_history[-1])
    _plot_accuracy(412, train_acc_history, test_acc_history, "max_depth",
                   max_depth_range)
    if best_acc < np.max(test_acc_history):
        best_acc = np.max(test_acc_history)
        best_params = {
            "max_depth": max_depth_range[np.argmax(test_acc_history)]
        }
    print("best params for single tree:", best_params)

    # find best random forest

    train_acc_history = []
    test_acc_history = []
    for n_estimators in n_estimators_range:
        forest = RandomForestClassifier(random_state=47,
                                        n_jobs=-1,
                                        n_estimators=n_estimators,
                                        **best_params)
        forest.fit(X_train, y_train)
        train_acc_history.append(forest.score(X_train, y_train))
        test_acc_history.append(forest.score(X_test, y_test))
        print("n_estimators:", n_estimators, "\ttrain_acc:",
              train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
    _plot_accuracy(413, train_acc_history, test_acc_history, "n_estimators",
                   n_estimators_range)
    best_acc = np.max(test_acc_history)
    best_params['n_estimators'] = n_estimators_range[np.argmax(
        test_acc_history)]

    train_acc_history = []
    test_acc_history = []
    for max_samples in max_samples_range:
        forest = RandomForestClassifier(random_state=47,
                                        n_jobs=-1,
                                        max_samples=max_samples,
                                        **best_params)
        forest.fit(X_train, y_train)
        train_acc_history.append(forest.score(X_train, y_train))
        test_acc_history.append(forest.score(X_test, y_test))
        print("max_samples:", max_samples, "\ttrain_acc:",
              train_acc_history[-1], "\ttest_acc:", test_acc_history[-1])
    _plot_accuracy(414, train_acc_history, test_acc_history, "max_samples",
                   max_samples_range)
    best_acc = np.max(test_acc_history)
    best_params['max_samples'] = max_samples_range[np.argmax(test_acc_history)]

    print("best params:", best_params, "best accuracy:", best_acc)
    plt.show()
