import pandas as pd
from sklearn import datasets


def read_iris_ds() -> pd.DataFrame:
    return pd.read_csv('data/Iris.csv').iloc[:, 1:5]


# Returns only generated samples without class membership - in order
# to do this return tuple of samples and classes
def read_moons_ds() -> pd.DataFrame:
    samples, classes = datasets.make_moons(n_samples=1000, noise=0.09, random_state=1)
    return pd.DataFrame(samples, columns=["X", "Y"])
