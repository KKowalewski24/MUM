import pandas as pd
from sklearn.impute import KNNImputer
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

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
    df_to_regression_model = df.dropna(subset = headers)
    df_to_regression_model = df_to_regression_model.loc[:, headers]
    del(headers[0])
    age_x = df_to_regression_model[headers]
    age_y = df_to_regression_model['age']

    age_lm = LinearRegression().fit(age_x, age_y)

    temp = interpolate(df)
    temp['age'] = df['age']

    temp[temp.isnull().any(axis=1)] #zostawia tylko wiersze z jakims nullem
    del temp['age']
    predicted_age = age_lm.predict(temp)
    df['age'] = df['age'].fillna(
        pd.Series(
            predicted_age[
                :df['age'].isna().sum()
                ], index=df.index[
                    df['age'].isna()
                    ][:len(predicted_age)]
            )
        )

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)


    return df



##regresja liniowa, logistyczna do płci/target, (też logistic linear, ale z multiclass taki argument) dla takich z kilkoma wartościami
##scilearn => 

##1wywalam braki, z okroojonego zbioru model regresji zrobić dla każdej kolumny
##potem imputacja dla każdej kolumny według modelu regresji jak brakuje danych w inne kolumnie to tymczasowo
## uzupełniasz danymi z innej metody imputacji