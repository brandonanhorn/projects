import pandas as pd

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

%matplotlib inline

df = pd.read_csv('data/heart.csv')

df.head()
df['target'].value_counts()

target = 'target'
y = df[target]
X = df.drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)
