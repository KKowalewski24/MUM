from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

from module.latex_generator import generate_image_figure, generate_table


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

    for i in range(2, 13):
        calculate_regression(df, 0, i, save_tables, missing_values_level, description)
        calculate_regression(df, 1, i, save_tables, missing_values_level, description)


def calculate_mean(df: pd.DataFrame, save_tables: bool,
                   missing_values_level: str, description: str) -> None:
    statistic_type = "Mean"
    mean = df.mean()
    display_result(statistic_type, mean)
    if save_tables:
        generate_table(
            mean.index, mean.values,
            create_table_filename(missing_values_level, description, statistic_type)
        )


def calculate_std(df: pd.DataFrame, save_tables: bool,
                  missing_values_level: str, description: str) -> None:
    statistic_type = "Standard Deviation"
    std = df.std()
    display_result(statistic_type, std)
    if save_tables:
        generate_table(
            std.index, std.values,
            create_table_filename(missing_values_level, description, statistic_type)
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
            create_table_filename(missing_values_level, description, statistic_type)
        )


def calculate_first_quantile(df: pd.DataFrame, save_tables: bool,
                             missing_values_level: str, description: str) -> None:
    statistic_type = "First quantile"
    quantile = df.quantile([0.25])
    display_result(statistic_type, quantile)
    if save_tables:
        generate_table(
            quantile.columns.tolist(), quantile.values[0].tolist(),
            create_table_filename(missing_values_level, description, statistic_type)
        )


def calculate_median(df: pd.DataFrame, save_tables: bool,
                     missing_values_level: str, description: str) -> None:
    statistic_type = "Median (Second quantile)"
    median = df.median()
    display_result(statistic_type, median)
    if save_tables:
        generate_table(
            median.index.tolist(), median.values.tolist(),
            create_table_filename(missing_values_level, description, statistic_type)
        )


def calculate_third_quantile(df: pd.DataFrame, save_tables: bool,
                             missing_values_level: str, description: str) -> None:
    statistic_type = "Third quantile"
    quantile = df.quantile([0.75])
    display_result(statistic_type, quantile)
    if save_tables:
        generate_table(
            quantile.columns.tolist(), quantile.values[0].tolist(),
            create_table_filename(missing_values_level, description, statistic_type)
        )


def calculate_regression(df: pd.DataFrame, y_axis_column_number: int,
                         x_axis_column_number: int, save_tables: bool,
                         missing_values_level: str, description: str) -> None:
    y_axis = df.iloc[:, y_axis_column_number].values.reshape(-1, 1)
    x_axis = df.iloc[:, x_axis_column_number].values.reshape(-1, 1)
    # TODO CHECK THIS WITH JANEK!!! MODE SOMETIMES RETURN EMPTY
    #  ARRAY THIS IS WHY WE HAVE TO CHECK
    if x_axis.shape[0] == 0:
        return

    linear_regression = LinearRegression()
    linear_regression.fit(x_axis, y_axis)
    y_axis_prediction = linear_regression.predict(x_axis)
    plt.scatter(x_axis, y_axis)
    plt.ylabel(replace_dash_with_space(df.columns[y_axis_column_number]))
    plt.xlabel(replace_dash_with_space(df.columns[x_axis_column_number]))
    plt.plot(x_axis, y_axis_prediction, color='red')
    print("Coefficient: ", linear_regression.coef_[0][0])

    if save_tables:
        filename = create_chart_filename(
            missing_values_level, description,
            df.columns[y_axis_column_number],
            df.columns[x_axis_column_number]
        )
        generate_image_figure(filename)
        plt.savefig(filename)
        plt.close()

    plt.show()


def display_separator() -> None:
    print("------------------------------------------------------------------------")


def display_result(description: str, result: Union[DataFrame, Series]) -> None:
    print("\n" + description + "\n", result)


def create_table_filename(missing_values_level: str, description: str,
                          statistic_type: str) -> str:
    return "result_" + missing_values_level + "_" \
           + replace_space_with_dash(description) + "_" \
           + replace_space_with_dash(statistic_type)


def create_chart_filename(missing_values_level: str, description: str,
                          first_column_name: str, second_column_name: str) -> str:
    return "regression-" + missing_values_level + "-" \
           + replace_space_with_dash(description) + "-" \
           + replace_space_with_dash(first_column_name) + "-" \
           + replace_space_with_dash(second_column_name)


def replace_space_with_dash(value: str) -> str:
    return value.replace(" ", "-")


def replace_dash_with_space(value: str) -> str:
    return value.replace("-", " ")


class MissingValuesException(Exception):
    pass
