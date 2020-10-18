import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("data/clean_data.csv")
del df["Unnamed: 0"]

df.head(1)

target = 'water_temperature'
X = df.drop(target, axis=1)
y = df[target]

y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    ("beach_name", LabelBinarizer()),
    (['turbidity'], StandardScaler()),
    (['wave_height'], StandardScaler()),
    (['wave_period'], StandardScaler()),
    (['battery_life'], StandardScaler()),
    ('time_of_day', LabelBinarizer()),
    (['month'], StandardScaler()),
    (['day'], StandardScaler()),
    (['year'], StandardScaler())],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = LinearRegression()
model.fit(Z_train, y_train)
print(model.score(Z_train, y_train), " - Linear Regression")
print(model.score(Z_test, y_test))

model = Lasso()
model.fit(Z_train, y_train)
print(model.score(Z_train, y_train), " - Lasso")
print(model.score(Z_test, y_test))

model = Ridge()
model.fit(Z_train, y_train)
print(model.score(Z_train, y_train), " - Ridge")
print(model.score(Z_test, y_test))

model = HuberRegressor()
model.fit(Z_train, y_train)
print(model.score(Z_train, y_train), " - Huber Regressor")
print(model.score(Z_test, y_test))

model = ElasticNet()
model.fit(Z_train, y_train)
print(model.score(Z_train, y_train), " - Elastic Net")
print(model.score(Z_test, y_test))

model = DecisionTreeRegressor()
model.fit(Z_train, y_train)
print(model.score(Z_train, y_train), " - Decision Tree")
print(model.score(Z_test, y_test))

model = RandomForestRegressor()
model.fit(Z_train, y_train)
print(model.score(Z_train, y_train), " - Random Forest")
print(model.score(Z_test, y_test))
 
