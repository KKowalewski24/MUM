import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


def read_iris_ds() -> pd.DataFrame:
    return pd.read_csv("data/Iris.csv").iloc[:, 1:5]


def read_mall_customers() -> pd.DataFrame:
    return pd.read_csv("data/Mall_Customers.csv").iloc[:, 1:5]


# Returns only generated samples without class membership - in order
# to do this return tuple of samples and classes
def read_moons_ds() -> pd.DataFrame:
    samples, classes = datasets.make_moons(n_samples=1000, noise=0.09, random_state=1)
    return pd.DataFrame(samples, columns=["X", "Y"])


def present_data_sets() -> None:
    irises: pd.DataFrame = read_iris_ds()
    plt.scatter(irises.iloc[:, 0], irises.iloc[:, 1])
    plt.title("Iris Sepal")
    plt.show()

    plt.scatter(irises.iloc[:, 2], irises.iloc[:, 3])
    plt.title("Iris Petal")
    plt.show()

    mall_customers: pd.DataFrame = read_mall_customers()
    plt.scatter(mall_customers.iloc[:, 0], mall_customers.iloc[:, 3])
    plt.title("Customers Gender")
    plt.show()

    plt.scatter(mall_customers.iloc[:, 1], mall_customers.iloc[:, 3])
    plt.title("Customers Age")
    plt.show()

    plt.scatter(mall_customers.iloc[:, 2], mall_customers.iloc[:, 3])
    plt.title("Customers Annual income")
    plt.show()

    moons: pd.DataFrame = read_moons_ds()
    plt.scatter(moons.iloc[:, 0], moons.iloc[:, 1])
    plt.title("Moons")
    plt.show()
