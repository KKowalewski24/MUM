from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

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

    filename = ""
    if save_tables:
        filename = "regression_" + missing_values_level + "_" + description
    calculate_regression(df, 0, [3, 4], filename)


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
    # TODO CHECK THIS WITH JANEK!!!
    if save_tables and len(mode.values) > 0:
        generate_table(
            mode.columns.tolist(), mode.values[0].tolist(),
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_first_quantile(df: pd.DataFrame, save_tables: bool,
                             missing_values_level: str, description: str) -> None:
    statistic_type = "First quantile"
    quantile = df.quantile([0.25])
    display_result(statistic_type, quantile)
    if save_tables:
        generate_table(
            quantile.columns.tolist(), quantile.values[0].tolist(),
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_median(df: pd.DataFrame, save_tables: bool,
                     missing_values_level: str, description: str) -> None:
    statistic_type = "Median (Second quantile)"
    median = df.median()
    display_result(statistic_type, median)
    if save_tables:
        generate_table(
            median.index.tolist(), median.values.tolist(),
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_third_quantile(df: pd.DataFrame, save_tables: bool,
                             missing_values_level: str, description: str) -> None:
    statistic_type = "Third quantile"
    quantile = df.quantile([0.75])
    display_result(statistic_type, quantile)
    if save_tables:
        generate_table(
            quantile.columns.tolist(), quantile.values[0].tolist(),
            create_filename(missing_values_level, description, statistic_type)
        )


def calculate_regression(df: pd.DataFrame, y_axis_column_number: int,
                         x_axis_column_numbers: Union[int, List[int]],
                         filename: str = "") -> None:
    y_axis = df.iloc[:, y_axis_column_number].values.reshape(-1, 1)
    x_axis = df.iloc[:, x_axis_column_numbers].values.reshape(-1, 1)
    # TODO CHECK THIS WITH JANEK!!! MODE SOMETIMES RETURN EMPTY
    #  ARRAY THIS IS WHY WE HAVE TO CHECK
    if x_axis.shape[0] == 0:
        return

    linear_regression = LinearRegression()
    linear_regression.fit(x_axis, get_concatenated_y_axis_array(y_axis, len(x_axis)))
    y_axis_prediction = linear_regression.predict(x_axis)
    plt.scatter(x_axis, get_concatenated_y_axis_array(y_axis, len(x_axis)))
    plt.plot(x_axis, y_axis_prediction, color='red')
    print("Coefficient: ", linear_regression.coef_)

    if filename != "":
        plt.savefig(filename)
    plt.show()


def get_concatenated_y_axis_array(y_axis: np.ndarray, x_axis_len: int) -> np.ndarray:
    return np.concatenate([y_axis for _ in range(int(x_axis_len / len(y_axis)))])


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
