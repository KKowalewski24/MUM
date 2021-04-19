from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

test_set_size = 0.3


def read_heart_ds() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv('data/heart.csv')
    X = df.drop('target', axis=1).to_numpy()
    y = df['target'].to_numpy()
    return train_test_split(X, y, test_size=test_set_size)


def read_gestures_ds() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv('data/gestures.csv')
    X = df.iloc[:, 0:63].to_numpy()
    y = df.iloc[:, 64].to_numpy()
    return train_test_split(X, y, test_size=test_set_size)


def read_weather_AUS() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # split train test
    X = df.drop('RainTomorrow', axis=1).to_numpy()
    y = df['RainTomorrow'].to_numpy()
    return train_test_split(X, y, test_size=test_set_size)
