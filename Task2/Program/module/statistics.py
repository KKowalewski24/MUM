import pandas as pd


def calculate_statistics(df: pd.DataFrame) -> None:
    if df.isna().sum().sum() != 0:
        print(df.isna())
        raise MissingValuesException

    print("Średnia\n", df.mean())


class MissingValuesException(Exception):
    pass
