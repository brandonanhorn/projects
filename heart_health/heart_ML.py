import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline

df = pd.read_csv('data/heart.csv')

df.info()
df.describe().T

target = 'target'
X = df.drop(target, axis=1)
y = df[target]

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

model = LinearRegression()
model.fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)


model = Lasso()
model.fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)

model = RandomForestRegressor()
model.fit(Z_train, y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)
