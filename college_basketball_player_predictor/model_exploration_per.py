import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
del df["vorp"]

target = "per"
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


## trying to maximize RandomForestRegressor
#model = RandomForestRegressor()
#params = {
#    "n_estimators": [25, 50, 100, 150, 175],
#    "criterion": ["mse", "mae"],
#    "max_depth": [0, 2, 4, 6, 8],
#    "min_samples_split": [0, 1, 2, 3, 4]
#}
#
#grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)
#
#grid.fit(Z_train, y_train)
#grid.best_score_
#grid.best_params_
#
#model = RandomForestRegressor()
#params = {
#    "n_estimators": [30, 40, 50, 60, 70],
#    "criterion": ["mae"],
#    "max_depth": [8, 10, 12],
#    "min_samples_split": [4, 5, 6]
#}
#grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)
#
#grid.fit(Z_train, y_train)
#grid.best_score_
#grid.best_params_

## best produced model
model = RandomForestRegressor(n_estimators = 50, criterion = "mae",
                            max_depth = 8, min_samples_split = 5).fit(Z_train,y_train)

print("RandomForestRegressor train score is " + str(model.score(Z_train,y_train)))
print("RandomForestRegressor test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))
