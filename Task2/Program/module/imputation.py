import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


def mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean())


def interpolate(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate().dropna()


def hot_deck(df: pd.DataFrame) -> pd.DataFrame:
    inputer = KNNImputer(n_neighbors=1)
    inserted_df = pd.DataFrame(inputer.fit_transform(df))
    inserted_df.columns = df.columns
    return inserted_df


def regression(df: pd.DataFrame) -> pd.DataFrame:
    headers = df.columns.tolist()
    discreet_headers = [headers[0], headers[3], headers[4], headers[7], headers[9]]
    not_discreet_headers = [headers[1], headers[2], headers[5], headers[6], headers[8], headers[10],
                         headers[11], headers[12], headers[13]]
    i = 0
    for header in headers:
        df_to_regression_model = df.dropna()
        headers.remove(header)
        x = df_to_regression_model[headers].to_numpy()
        y = df_to_regression_model[header].to_numpy()

        if len(y) < 2 :
            print("Not enough data to create linear regression model.")
            return pd.DataFrame()

        if header in discreet_headers:
            lm = LinearRegression().fit(x, y)
        elif header in not_discreet_headers:
            if len(np.unique(y)) < 2: 
                print("Not enough data to create logistic regression model.")
                return pd.DataFrame()
            lm = LogisticRegression(max_iter=1000000000000).fit(x, y)

        temp = interpolate(df)
        temp[header] = df[header]

        temp[temp.isnull().any(axis=1)]  # zostawia tylko wiersze z jakims nullem

        del temp[header]
        predicted = lm.predict(temp)

        df[header] = df[header].fillna(
            pd.Series(
                predicted[
                :df[header].isna().sum()
                ], index=df.index[
                             df[header].isna()
                         ][:len(predicted)]
            )
        )  # to skomplikowane przyrownanie zastepuje w df jedna kolumne z danymi wlasnie "przewidzianymi" danymi
        headers.insert(i, header)
        i = i + 1

    return df