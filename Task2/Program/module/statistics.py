from typing import Union

import pandas as pd
from pandas import DataFrame, Series

from module.table_generator import generate_table


def calculate_statistics(df: pd.DataFrame, save_tables: bool,
                         missing_values_level: str, description: str) -> None:
    if df.isna().sum().sum() != 0:
        print(df.isna())
        raise MissingValuesException

    display_separator()
    print(description)
    calculate_mean(df, save_tables, missing_values_level, description)
    calculate_std(df, save_tables, missing_values_level, description)
    calculate_mode(df, save_tables, missing_values_level, description)
    calculate_first_quantile(df, save_tables, missing_values_level, description)
    calculate_median(df, save_tables, missing_values_level, description)
    calculate_third_quantile(df, save_tables, missing_values_level, description)


def calculate_mean(df: pd.DataFrame, save_tables: bool,
                   missing_values_level: str, description: str) -> None:
    statistic_type = "Mean"
    mean = df.mean()
    display_result(statistic_type, mean)
    if save_tables:
        generate_table(
            mean.index, mean.values,
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_std(df: pd.DataFrame, save_tables: bool,
                  missing_values_level: str, description: str) -> None:
    statistic_type = "Standard Deviation"
    std = df.std()
    display_result(statistic_type, std)
    if save_tables:
        generate_table(
            std.index, std.values,
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_mode(df: pd.DataFrame, save_tables: bool,
                   missing_values_level: str, description: str) -> None:
    statistic_type = "Mode"
    mode = df.mode()
    display_result(statistic_type, mode)
    if save_tables:
        generate_table(
            mode.index, mode.values,
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_first_quantile(df: pd.DataFrame, save_tables: bool,
                             missing_values_level: str, description: str) -> None:
    statistic_type = "First quantile"
    quantile = df.quantile([0.25])
    display_result(statistic_type, quantile)
    if save_tables:
        generate_table(
            quantile.index, quantile.values,
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_median(df: pd.DataFrame, save_tables: bool,
                     missing_values_level: str, description: str) -> None:
    statistic_type = "Median (Second quantile)"
    median = df.median()
    display_result(statistic_type, median)
    if save_tables:
        generate_table(
            median.index, median.values,
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_third_quantile(df: pd.DataFrame, save_tables: bool,
                             missing_values_level: str, description: str) -> None:
    statistic_type = "Third quantile"
    quantile = df.quantile([0.75])
    display_result(statistic_type, quantile)
    if save_tables:
        generate_table(
            quantile.index, quantile.values,
            create_filename(missing_values_level, description, statistic_type)
        )


def display_separator() -> None:
    print("------------------------------------------------------------------------")


def display_result(description: str, result: Union[DataFrame, Series]) -> None:
    print("\n" + description + "\n", result)


def create_filename(missing_values_level: str, description: str,
                    statistic_type: str) -> str:
    return "result_" + missing_values_level + "_" \
           + replace_space_with_dash(description) + "_" \
           + replace_space_with_dash(statistic_type)


def replace_space_with_dash(value: str) -> str:
    return value.replace(" ", "-")


class MissingValuesException(Exception):
    pass
