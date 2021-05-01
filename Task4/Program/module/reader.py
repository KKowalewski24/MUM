import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


def read_iris_ds() -> np.ndarray:
    return pd.read_csv("data/Iris.csv").iloc[:, 1:5].to_numpy()


def read_mall_customers() -> np.ndarray:
    df = pd.read_csv("data/Mall_Customers.csv")
    df.iloc[:, 1] = LabelEncoder().fit_transform(df.iloc[:, 1])
    return df.iloc[:, 1:5].to_numpy()


# Returns only generated samples without class membership - in order
# to do this return tuple of samples and classes
def read_moons_ds() -> np.ndarray:
    samples, classes = datasets.make_moons(n_samples=700, noise=0.09, random_state=1)
    return samples


def present_data_sets() -> None:
    irises: np.ndarray = read_iris_ds()
    plt.scatter(irises[:, 0], irises[:, 1])
    plt.title("Iris Sepal")
    plt.show()

    plt.scatter(irises[:, 2], irises[:, 3])
    plt.title("Iris Petal")
    plt.show()

    mall_customers: np.ndarray = read_mall_customers()
    plt.scatter(mall_customers[:, 0], mall_customers[:, 3])
    plt.title("Customers Gender")
    plt.show()

    plt.scatter(mall_customers[:, 1], mall_customers[:, 3])
    plt.title("Customers Age")
    plt.show()

    plt.scatter(mall_customers[:, 2], mall_customers[:, 3])
    plt.title("Customers Annual income")
    plt.show()

    moons: np.ndarray = read_moons_ds()
    plt.scatter(moons[:, 0], moons[:, 1])
    plt.title("Moons")
    plt.show()
