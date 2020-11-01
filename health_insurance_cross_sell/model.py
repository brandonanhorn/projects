import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn_pandas import DataFrameMapper

from catboost import CatBoostRegressor

df = pd.read_csv("data/train.csv")
pd.set_option("max.columns", None)

df.head(1)

target = "Response"
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    ("Gender", LabelBinarizer()),
    (["Age"], StandardScaler()),
    (["Driving_License"], StandardScaler()),
    (["Region_Code"], StandardScaler()),
    (["Previously_Insured"], StandardScaler()),
    ("Vehicle_Age", LabelBinarizer()),
    ("Vehicle_Damage", LabelBinarizer()),
    (["Annual_Premium"], StandardScaler()),
    (["Policy_Sales_Channel"], StandardScaler()),
    (["Vintage"], StandardScaler())],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = CatBoostRegressor(iterations=5000).fit(Z_train,y_train)
print("CatBoostRegressor train score is " + str(model.score(Z_train,y_train)))
print("CatBoostRegressor test score is " + str(model.score(Z_test,y_test)))

test1 = pd.read_csv('data/test.csv')

test1['Response'] = model.predict(mapper.transform(test1))
test1

test1[['id','Response']].to_csv('test1.csv', index=False)
