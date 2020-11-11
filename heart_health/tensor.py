import pandas as pd

import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical

%matplotlib inline

df = pd.read_csv('data/heart.csv')

df.head()
df['target'].value_counts()

target = 'target'
y = df[target]
X = df.drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    (['age'], StandardScaler()),
    ('sex', LabelBinarizer()),
    ('cp', LabelBinarizer()),
    (['trestbps'], StandardScaler()),
    (['chol'], StandardScaler()),
    ('fbs', LabelBinarizer()),
    ('restecg', LabelBinarizer()),
    (['thalach'], StandardScaler()),
    ('exang', LabelBinarizer()),
    (['oldpeak'], StandardScaler()),
    ('slope', LabelBinarizer()),
    ('ca', LabelBinarizer()),
    ('thal', LabelBinarizer())],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = Sequential([
    Input(shape=(Z_train.shape[1],)),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='elu')
])

model.compile(loss='mae', optimizer='adam')

model.fit(Z_train, y_train, epochs=12,
            batch_size=32, validation_data=(Z_test, y_test))

model.evaluate(Z_test, y_test)
