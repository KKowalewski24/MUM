import pandas as pd


def calculate_statistics(df: pd.DataFrame) -> None:
    if df.isna().sum().sum() != 0:
        print(df.isna())
        raise MissingValuesException

    print("\nMean\n", df.mean())
    print("\nStandard Deviation\n", df.std())
    print("\nMode\n", df.mode())
    print("\nFirst quantile\n", df.quantile([0.25]))
    print("\nMedian (Second quantile)\n", df.median())
    print("\nThird quantile\n", df.quantile([0.75]))


class MissingValuesException(Exception):
    pass
