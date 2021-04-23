import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_heart_ds() -> pd.DataFrame:
    return pd.read_csv('data/heart.csv')


def read_gestures_ds() -> pd.DataFrame:
    return pd.read_csv('data/gestures.csv')


def read_weather_AUS_ds() -> pd.DataFrame:
    df = pd.read_csv('data/weatherAUS.csv').dropna()

    # encode date as day-of-year
    df['DayOfYear'] = pd.to_datetime(
        df['Date']).map(lambda date: date.day_of_year)
    df.drop('Date', axis=1, inplace=True)

    # encode categorical columns
    categorical_columns = [
        'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'
    ]
    encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    return df
