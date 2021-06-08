from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_letters_ds() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    letters_train = pd.read_csv('data/isolet_train.csv', header=None)
    X_train = letters_train.iloc[:, :-1].to_numpy()
    y_train = letters_train.iloc[:, -1].to_numpy()

    letters_test = pd.read_csv('data/isolet_test.csv', header=None)
    X_test = letters_test.iloc[:, :-1].to_numpy()
    y_test = letters_test.iloc[:, -1].to_numpy()

    return X_train, X_test, y_train, y_test


def read_numerals_ds() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numerals = pd.read_csv('data/numerals.csv', header=None)
    X = numerals.iloc[:, :-1].to_numpy()
    y = numerals.iloc[:, -1].to_numpy()
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=47)


def read_documents_ds() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    documents = pd.read_csv('data/documents.csv', header=None)
    X = documents.iloc[:, 1:].to_numpy()
    y = documents.iloc[:, 0].to_numpy()
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=47)
