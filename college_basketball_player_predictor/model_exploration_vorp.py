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
