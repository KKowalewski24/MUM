import pandas as pd
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
    headers = [
        'age','sex','chest-pain-type','resting-blood-pressure','serum-cholestoral','fasting-blood-sugar','resting-electrocardiographic-results','maximum-heart-rate','exercise-induced-angina','oldpeak','the-slope-of-the-peak-exercise','number-of-major-vessels','thal','target'
    ]
    fuzzy_headers = [ 'age','resting-blood-pressure','serum-cholestoral', 'maximum-heart-rate','oldpeak']
    not_fuzzy_headers = ['sex','chest-pain-type','fasting-blood-sugar','resting-electrocardiographic-results','exercise-induced-angina','the-slope-of-the-peak-exercise','number-of-major-vessels','thal','target']
    i = 0
    # [ 'age','resting-blood-pressure','serum-cholestoral, 'maximum-heart-rate','oldpeak']
    for header in headers:
        df_to_regression_model = df.dropna(subset = headers)
        df_to_regression_model = df_to_regression_model.loc[:, headers]
        headers.remove(header)
        x = df_to_regression_model[headers]
        y = df_to_regression_model[header]

        if header in fuzzy_headers:
            lm = LinearRegression().fit(x, y)
        elif header in not_fuzzy_headers:
            lm = LogisticRegression(max_iter=1000000000000).fit(x, y)


        temp = interpolate(df)
        temp[header] = df[header]

        temp[temp.isnull().any(axis=1)] #zostawia tylko wiersze z jakims nullem
        # is_NaN = temp.isnull()
        # row_has_NaN = is_NaN.any(axis=1)
        # temp = temp[row_has_NaN]


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
            ) #to skomplikowane przyrownanie zastepuje w df jedna kolumne z danymi wlasnie "przewidzianymi" danymi
        headers.insert(i, header)
        i = i + 1
        


    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)


    return df



##regresja liniowa, logistyczna do płci/target, (też logistic linear, ale z multiclass taki argument) dla takich z kilkoma wartościami
##scilearn => 

##1wywalam braki, z okroojonego zbioru model regresji zrobić dla każdej kolumny
##potem imputacja dla każdej kolumny według modelu regresji jak brakuje danych w inne kolumnie to tymczasowo
## uzupełniasz danymi z innej metody imputacji