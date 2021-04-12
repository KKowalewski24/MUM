from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

from module.latex_generator import RESULTS_DIR_NAME, generate_image_figure, generate_table


def calculate_statistics(df: pd.DataFrame, save_tables: bool,
                         missing_values_level: str, description: str) -> None:
    if df.isna().sum().sum() != 0:
        print(df.isna())
        raise MissingValuesException

    display_separator()
    print(description)

    if df.empty:
        print("Empty data frame, no way to calculate something!")
        return

    basic_statistics = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std(),
        'Mode': df.mode().iloc[0],
        'Q1': df.quantile(0.25),
        'Median': df.median(),
        'Q2': df.quantile(0.75)
    })

    print(basic_statistics)

    if save_tables:
        generate_table(
            basic_statistics,
            create_table_filename(missing_values_level, description)
        )

    calculate_regression(df, 0, 3, save_tables, missing_values_level, description)
    calculate_regression(df, 0, 7, save_tables, missing_values_level, description)


def calculate_regression(df: pd.DataFrame, y_axis_column_number: int,
                         x_axis_column_number: int, save_tables: bool,
                         missing_values_level: str, description: str) -> None:
    y = df.iloc[:, y_axis_column_number].values.reshape(-1, 1)
    x = df.iloc[:, x_axis_column_number].values.reshape(-1, 1)

    linear_regression = LinearRegression()
    linear_regression.fit(x, y)
    y_pred = linear_regression.predict(x)
    plt.scatter(x, y)
    plt.ylabel(replace_dash_with_space(df.columns[y_axis_column_number]))
    plt.xlabel(replace_dash_with_space(df.columns[x_axis_column_number]))
    plt.plot(x, y_pred, color='red')
    regression_coef = round(linear_regression.coef_[0][0], 4)
    regression_intercept = round(linear_regression.intercept_[0], 4)
    print("Coefficient: ", regression_coef)
    print("Intercept: ", regression_intercept)

    if save_tables:
        filename = create_chart_filename(missing_values_level, description,
                                         df.columns[y_axis_column_number],
                                         df.columns[x_axis_column_number])
        generate_image_figure(filename, regression_coef, regression_intercept)
        plt.savefig(RESULTS_DIR_NAME + "/" + filename)
        plt.close()

    plt.show()


def display_separator() -> None:
    print(
        "------------------------------------------------------------------------"
    )


def display_result(description: str, result: Union[DataFrame, Series]) -> None:
    print("\n" + description + "\n", result)


def create_table_filename(missing_values_level: str, description: str) -> str:
    return "result_" + missing_values_level + "_" \
           + replace_space_with_dash(description)


def create_chart_filename(missing_values_level: str, description: str,
                          first_column_name: str,
                          second_column_name: str) -> str:
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
