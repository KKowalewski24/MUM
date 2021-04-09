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
    age_x = df_to_regression_model[headers[1:]] #moze dodatkowy nawias
    age_y = df_to_regression_model['age']

    # age_x_train, age_x_test, age_y_train, age_y_test = train_test_split(age_x, age_y, test_size=0.2, random_state=101)
    age_lm = LinearRegression().fit(age_x, age_y)

    temp = interpolate(df)
    temp['age'] = df['age']

    is_NaN = temp.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    temp = temp[row_has_NaN]

    # missing = df['age'].isnull()
    # nex = pd.DataFrame(df[headers][missing])


    print("temp ")
    del temp['age']
    print(temp)

    # age_lm.predict(nex)

    pred = age_lm.predict(temp)
    temp.insert(0, 'Age', pred)
    print('zrobione')
    print(temp)
    return df



##regresja liniowa, logistyczna do płci/target, (też logistic linear, ale z multiclass taki argument) dla takich z kilkoma wartościami
##scilearn => 

##1wywalam braki, z okroojonego zbioru model regresji zrobić dla każdej kolumny
##potem imputacja dla każdej kolumny według modelu regresji jak brakuje danych w inne kolumnie to tymczasowo
## uzupełniasz danymi z innej metody imputacji