import pandas as pd


def calculate_statistics(df: pd.DataFrame, save_tables: bool,
                         missing_values_level: str, description: str) -> None:
    if df.isna().sum().sum() != 0:
        print(df.isna())
        raise MissingValuesException

    display_separator()
    print(description)
    calculate_mean(df, save_tables)
    calculate_std(df, save_tables)
    calculate_mode(df, save_tables)
    calculate_first_quantile(df, save_tables)
    calculate_median(df, save_tables)
    calculate_third_quantile(df, save_tables)


def calculate_mean(df: pd.DataFrame, save_tables: bool):
    mean = df.mean()
    print("\nMean\n", mean)


def calculate_std(df: pd.DataFrame, save_tables: bool):
    std = df.std()
    print("\nStandard Deviation\n", std)


def calculate_mode(df: pd.DataFrame, save_tables: bool):
    mode = df.mode()
    print("\nMode\n", mode)


def calculate_first_quantile(df: pd.DataFrame, save_tables: bool):
    quantile = df.quantile([0.25])
    print("\nFirst quantile\n", quantile)


def calculate_median(df: pd.DataFrame, save_tables: bool):
    median = df.median()
    print("\nMedian (Second quantile)\n", median)


def calculate_third_quantile(df: pd.DataFrame, save_tables: bool):
    quantile = df.quantile([0.75])
    print("\nThird quantile\n", quantile)


def display_separator() -> None:
    print("------------------------------------------------------------------------")


class MissingValuesException(Exception):
    pass
