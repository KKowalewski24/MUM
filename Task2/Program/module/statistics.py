import pandas as pd


def calculate_statistics(df: pd.DataFrame, save_tables: bool) -> None:
    if df.isna().sum().sum() != 0:
        print(df.isna())
        raise MissingValuesException

    calculate_mean(df)
    calculate_std(df)
    calculate_mode(df)
    calculate_first_quantile(df)
    calculate_median(df)
    calculate_third_quantile(df)


def calculate_mean(df):
    print("\nMean\n", df.mean())


def calculate_std(df):
    print("\nStandard Deviation\n", df.std())


def calculate_mode(df):
    print("\nMode\n", df.mode())


def calculate_first_quantile(df):
    print("\nFirst quantile\n", df.quantile([0.25]))


def calculate_median(df):
    print("\nMedian (Second quantile)\n", df.median())


def calculate_third_quantile(df):
    print("\nThird quantile\n", df.quantile([0.75]))


class MissingValuesException(Exception):
    pass
