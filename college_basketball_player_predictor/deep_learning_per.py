import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/nba_college_players.csv")
df["height"] = df['height'].str.replace('"', "")
df["college_country"] = df["college_country"].astype("str")
df["position"] = df["position"].astype("str")
df["height"] = df["college_country"].astype("str")
df["handedness"] = df["college_country"].astype("str")

df.head(2)
del df["name"]
del df["vorp"]

target = "per"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    ("college_country", LabelBinarizer()),
    (["g"], StandardScaler()),
    (["gs"], StandardScaler()),
    (["mp"], StandardScaler()),
    (["fg"], StandardScaler()),
    (["fga"], StandardScaler()),
    (["fg%"], StandardScaler()),
    (["2p"], StandardScaler()),
    (["2pa"], StandardScaler()),
    (["2p%"], StandardScaler()),
    (["3p"], StandardScaler()),
    (["3pa"], StandardScaler()),
    (["3p%"], StandardScaler()),
    (["ft"], StandardScaler()),
    (["fta"], StandardScaler()),
    (["ft%"], StandardScaler()),
    (["orb"], StandardScaler()),
    (["drb"], StandardScaler()),
    (["trb"], StandardScaler()),
    (["ast"], StandardScaler()),
    (["stl"], StandardScaler()),
    (["blk"], StandardScaler()),
    (["tov"], StandardScaler()),
    (["pf"], StandardScaler()),
    (["pts"], StandardScaler()),
    (["sos"], StandardScaler()),
    ("position", LabelBinarizer()),
    ("height", LabelBinarizer()),
    ("handedness", LabelBinarizer())],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = LinearRegression().fit(Z_train,y_train)
model.score(Z_train,y_train)
model.score(Z_test,y_test)

## Model
model = Sequential([
    Input(shape=(Z_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(.01),
    Dense(16, activation='softmax'),
    Dropout(.01),
    Dense(1, activation='linear')
])

model.compile(loss='mae', optimizer='adam', metrics=["accuracy"])

history = model.fit(Z_train, y_train,
                    validation_data=(Z_test, y_test),
                    epochs=25, batch_size=128,
                    verbose=2)

model.summary()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend();

## Deep learning doesn't help
