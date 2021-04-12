import pandas as pd
import numpy as np
from typing import List
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression


def _fix_categorical_columns(original_df: pd.DataFrame,
                             imputated_df: pd.DataFrame,
                             categorical_columns: List[int]) -> None:
    for categorical_column in categorical_columns:
        values = np.unique(
            original_df.iloc[:, categorical_column].dropna().to_numpy())
        column = imputated_df.iloc[:, categorical_column].to_numpy()
        column = values[np.argmin(
            np.abs(np.reshape(column, (-1, 1)) - np.reshape(values, (1, -1))),
            axis=1)]
        imputated_df.iloc[:, categorical_column] = column


def mean(df: pd.DataFrame, categorical_columns: List[int]) -> pd.DataFrame:
    imputated_df = df.fillna(df.mean())
    _fix_categorical_columns(df, imputated_df, categorical_columns)
    return imputated_df


def interpolate(df: pd.DataFrame,
                categorical_columns: List[int]) -> pd.DataFrame:
    imputated_df = df.interpolate().dropna()
    _fix_categorical_columns(df, imputated_df, categorical_columns)
    return imputated_df


def hot_deck(df: pd.DataFrame) -> pd.DataFrame:
    inputer = KNNImputer(n_neighbors=1)
    inserted_df = pd.DataFrame(inputer.fit_transform(df))
    inserted_df.columns = df.columns
    return inserted_df


def regression(df: pd.DataFrame,
               categorical_columns: List[int]) -> pd.DataFrame:
    headers = df.columns.tolist()
    categorical_headers = df.columns[categorical_columns].tolist()
    i = 0
    for header in headers:
        df_to_regression_model = df.dropna()
        headers.remove(header)
        x = df_to_regression_model[headers].to_numpy()
        y = df_to_regression_model[header].to_numpy()

        if len(y) < 2:
            print("Not enough data to create linear regression model.")
            return pd.DataFrame()

        if header not in categorical_headers:
            lm = LinearRegression().fit(x, y)
        else:
            if len(np.unique(y)) < 2:
                print("Not enough data to create logistic regression model.")
                return pd.DataFrame()
            lm = LogisticRegression(solver='liblinear').fit(x, y)

        temp = mean(df, categorical_columns)
        temp[header] = df[header]

        temp[temp.isnull().any(
            axis=1)]  # zostawia tylko wiersze z jakims nullem

        del temp[header]
        predicted = lm.predict(temp)

        df[header] = df[header].fillna(
            pd.Series(predicted[:df[header].isna().sum()],
                      index=df.index[df[header].isna()][:len(predicted)])
        )  # to skomplikowane przyrownanie zastepuje w df jedna kolumne z danymi wlasnie "przewidzianymi" danymi
        headers.insert(i, header)
        i = i + 1

    return df
