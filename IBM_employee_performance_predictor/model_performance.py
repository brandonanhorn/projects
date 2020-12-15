import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/ibm_personal.csv")

df.columns = [columns.lower() for columns in df.columns]
df.head(1)

target = "performancerating"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    ("department", LabelBinarizer()),
    (["education"], StandardScaler()),
    ("educationfield", LabelBinarizer()),
    ("gender", LabelBinarizer()),
    ("jobrole", LabelBinarizer())],df_out=True)

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
