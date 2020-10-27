import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("data/clean_data.csv")
del df["Unnamed: 0"]
df["Embarked"] = df["Embarked"].astype(str)
df.head(1)
df["Age"] = df["Age"].fillna(0)

target = "Survived"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    (["Pclass"], StandardScaler()),
    ("Sex", LabelBinarizer()),
    (["Age"], StandardScaler()),
    (["SibSp"], StandardScaler()),
    (["Parch"], StandardScaler()),
    (["Fare"], StandardScaler()),
    ("Embarked", LabelBinarizer()),
    ("surnames", LabelBinarizer())
    ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = LinearRegression().fit(Z_train,y_train)
print("LinearRegression train score is " + str(model.score(Z_train,y_train)))
print("LinearRegression test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

model = HuberRegressor().fit(Z_train,y_train)
print("HuberRegressor train score is " + str(model.score(Z_train,y_train)))
print("HuberRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

model = Lasso().fit(Z_train,y_train)
print("Lasso train score is " + str(model.score(Z_train,y_train)))
print("Lasso test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

model = Ridge().fit(Z_train,y_train)
print("Ridge train score is " + str(model.score(Z_train,y_train)))
print("Ridge test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

model = ElasticNet().fit(Z_train,y_train)
print("ElasticNet train score is " + str(model.score(Z_train,y_train)))
print("ElasticNet test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

model = RandomForestRegressor().fit(Z_train,y_train)
print("RandomForestRegressor train score is " + str(model.score(Z_train,y_train)))
print("RandomForestRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))


model = DecisionTreeRegressor().fit(Z_train,y_train)
print("DecisionTreeRegressor train score is " + str(model.score(Z_train,y_train)))
print("DecisionTreeRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))


model = AdaBoostRegressor().fit(Z_train,y_train)
print("AdaBoostRegressor train score is " + str(model.score(Z_train,y_train)))
print("AdaBoostRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

model = CatBoostRegressor().fit(Z_train,y_train)
print("CatBoostRegressor train score is " + str(model.score(Z_train,y_train)))
print("CatBoostRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

model = Sequential([
    Input(shape=(Z_train.shape[1],)),
    Dense(50, activation='relu'),
    Dropout(.05),
    Dense(20, activation='relu'),
    Dropout(.05),
    Dense(1, activation='linear')
])

model.compile(loss='mae', optimizer='adam', metrics=["accuracy"])

history = model.fit(Z_train, y_train,
                    validation_data=(Z_test, y_test),
                    epochs=50, batch_size=32,
                    verbose=2)

model.summary()

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Test accuracy')
plt.legend();
