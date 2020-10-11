import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/nba_college_players.csv")
df["height"] = df['height'].str.replace('"', "")
df["college_country"] = df["college_country"].astype("str")
df["position"] = df["position"].astype("str")
df["height"] = df["college_country"].astype("str")
df["handedness"] = df["college_country"].astype("str")

df.head(2)
del df["name"]
del df["per"]

target = "vorp"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    ("college_country", LabelBinarizer()),
    (["g"], None),
    (["gs"], None),
    (["mp"], None),
    (["fg"], None),
    (["fga"], None),
    (["fg%"], None),
    (["2p"], None),
    (["2pa"], None),
    (["2p%"], None),
    (["3p"], None),
    (["3pa"], None),
    (["3p%"], None),
    (["ft"], None),
    (["fta"], None),
    (["ft%"], None),
    (["orb"], None),
    (["drb"], None),
    (["trb"], None),
    (["ast"], None),
    (["stl"], None),
    (["blk"], None),
    (["tov"], None),
    (["pf"], None),
    (["pts"], None),
    (["sos"], None),
    ("position", LabelBinarizer()),
    ("height", LabelBinarizer()),
    ("handedness", LabelBinarizer())],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

## model performance unusable
model = LinearRegression().fit(Z_train,y_train)
print("LinearRegression train score is " + str(model.score(Z_train,y_train)))
print("LinearRegression test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## model
model = HuberRegressor().fit(Z_train,y_train)
print("HuberRegressor train score is " + str(model.score(Z_train,y_train)))
print("HuberRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## model
model = Lasso().fit(Z_train,y_train)
print("Lasso train score is " + str(model.score(Z_train,y_train)))
print("Lasso test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## model is best performing
model = Ridge().fit(Z_train,y_train)
print("Ridge train score is " + str(model.score(Z_train,y_train)))
print("Ridge test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## model performs 2nd best
model = ElasticNet().fit(Z_train,y_train)
print("ElasticNet train score is " + str(model.score(Z_train,y_train)))
print("ElasticNet test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## model performs poorly
model = RandomForestRegressor().fit(Z_train,y_train)
print("RandomForestRegressor train score is " + str(model.score(Z_train,y_train)))
print("RandomForestRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## model performs very poorly
model = DecisionTreeRegressor().fit(Z_train,y_train)
print("DecisionTreeRegressor train score is " + str(model.score(Z_train,y_train)))
print("DecisionTreeRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## model performs poorly
model = AdaBoostRegressor().fit(Z_train,y_train)
print("AdaBoostRegressor train score is " + str(model.score(Z_train,y_train)))
print("AdaBoostRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## trying to maximize RandomForestRegressor
model = RandomForestRegressor()
params = {
    "n_estimators": [25, 50, 100, 150, 175],
    "criterion": ["mse", "mae"],
    "max_depth": [0, 2, 4, 6, 8],
    "min_samples_split": [0, 1, 2, 3, 4]
}

grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)

grid.fit(Z_train, y_train)
grid.best_score_
grid.best_params_
#
model = RandomForestRegressor()
params = {
    "n_estimators": [15, 20, 25, 30, 35],
    "criterion": ["mse", "mae"],
    "max_depth": [2, 4, 6],
    "min_samples_split": [2, 3, 4]
}
grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)

grid.fit(Z_train, y_train)
grid.best_score_
grid.best_params_

model = RandomForestRegressor()
params = {
    "n_estimators": [26, 28, 30, 32, 34],
    "criterion": ["mse", "mae"],
    "max_depth": [4, 5, 6, 7, 8],
    "min_samples_split": [3, 4, 5, 6]
}

grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)

grid.fit(Z_train, y_train)
grid.best_score_
grid.best_params_

## best produced RandomForestRegressor model
model = RandomForestRegressor(n_estimators = 30, criterion = "mae",
                            max_depth = 6, min_samples_split = 4).fit(Z_train,y_train)

print("RandomForestRegressor train score is " + str(model.score(Z_train,y_train)))
print("RandomForestRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))


## Trying to maximize Ridge(best performing model)
model = Ridge()
params = {
    "alpha": [0.25, 0.5, .75, 1, 1.25, 1.50, 1.75],
    "fit_intercept": [True, False],
    "normalize": [True, False],
    "copy_X": [True, False],
    "max_iter": [None, 1, 2, 3, 4, 5],
    "tol": [0.0005, 0.001, 0.002, 0.003]
}

grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)

grid.fit(Z_train, y_train)
grid.best_score_
grid.best_params_

model = Ridge(alpha = 1.75,copy_X = True, fit_intercept = True,
            max_iter = None, normalize = True, tol = 0.0005).fit(Z_train,y_train)
print("Ridge train score is " + str(model.score(Z_train,y_train)))
print("Ridge test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

## Lasso top performing model
