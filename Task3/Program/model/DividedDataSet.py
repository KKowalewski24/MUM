import pandas as pd


class DividedDataSet:

    def __init__(self, test_set_size: float, training_set: pd.DataFrame,
                 test_set: pd.DataFrame) -> None:
        self.test_set_size = test_set_size
        self.training_set = training_set
        self.test_set = test_set
