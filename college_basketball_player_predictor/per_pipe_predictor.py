import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import Ridge
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

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

model = Ridge().fit(Z_train,y_train)
print("Ridge train score is " + str(model.score(Z_train,y_train)))
print("Ridge test score is " + str(model.score(Z_test,y_test)))

print("Mean squared error is " + str(mean_squared_error(y_test, model.predict(Z_test))**(1/2)))

pipe = make_pipeline(mapper, model)

pipe.fit(X_train, y_train)

toppin = pd.DataFrame({
    'college_country': "Dayton",
    "g": [64],
    "gs": [46],
    "mp": [29],
    "fg": [7],
    "fga": [10.8],
    "fg%": [.647],
    "2p": [6.3],
    "2pa": [9.2],
    "2p%": [.688],
    "3p": [.7],
    "3pa": [1.6],
    "3p%": [.417],
    "ft": [2.5],
    "fta": [3.6],
    "ft%": [.706],
    "orb": [1.4],
    "drb": [5.2],
    "trb": [6.6],
    "ast": [2.0],
    "stl": [.8],
    "blk": [1.0],
    "tov": [2.0],
    "pf": [1.9],
    "pts": [17.1],
    "sos": [3.3],
    "position": "forward",
    "height": "6-9",
    "handedness": "right"}
)

# 13.44
pipe.predict(toppin)

edwards = pd.DataFrame({
    'college_country': "Georgia",
    "g": [32],
    "gs": [32],
    "mp": [33],
    "fg": [6.3],
    "fga": [15.8],
    "fg%": [.402],
    "2p": [4.1],
    "2pa": [8.1],
    "2p%": [.504],
    "3p": [2.3],
    "3pa": [7.7],
    "3p%": [.294],
    "ft": [4.1],
    "fta": [5.3],
    "ft%": [.772],
    "orb": [.8],
    "drb": [4.5],
    "trb": [5.2],
    "ast": [2.8],
    "stl": [1.3],
    "blk": [.6],
    "tov": [2.7],
    "pf": [2.2],
    "pts": [19.1],
    "sos": [8.06],
    "position": "guard",
    "height": "6-5",
    "handedness": "right"}
)

# 14.30
pipe.predict(edwards)

wiseman = pd.DataFrame({
    'college_country': "Memphis",
    "g": [3],
    "gs": [3],
    "mp": [23],
    "fg": [6.7],
    "fga": [8.7],
    "fg%": [.769],
    "2p": [6.7],
    "2pa": [8.3],
    "2p%": [.800],
    "3p": [0],
    "3pa": [.3],
    "3p%": [0],
    "ft": [6.3],
    "fta": [9.0],
    "ft%": [.704],
    "orb": [4.3],
    "drb": [6.3],
    "trb": [10.7],
    "ast": [.3],
    "stl": [.3],
    "blk": [3.0],
    "tov": [1.0],
    "pf": [1.7],
    "pts": [19.7],
    "sos": [5.07],
    "position": "center",
    "height": "7-1",
    "handedness": "left"}
)

# 27.31 - a bit shocking
pipe.predict(wiseman)

bey = pd.DataFrame({
    'college_country': "Villanova",
    "g": [67],
    "gs": [60],
    "mp": [31.6],
    "fg": [4.2],
    "fga": [9.0],
    "fg%": [.469],
    "2p": [2.3],
    "2pa": [4.5],
    "2p%": [.522],
    "3p": [1.9],
    "3pa": [4.6],
    "3p%": [.418],
    "ft": [1.5],
    "fta": [2.0],
    "ft%": [.728],
    "orb": [1.7],
    "drb": [3.2],
    "trb": [4.9],
    "ast": [1.8],
    "stl": [.8],
    "blk": [.4],
    "tov": [1.1],
    "pf": [2.3],
    "pts": [11.9],
    "sos": [9.00],
    "position": "forward",
    "height": "6-8",
    "handedness": "right"}
)

# 8.98
pipe.predict(bey)

okongwu = pd.DataFrame({
    'college_country': "USC",
    "g": [28],
    "gs": [28],
    "mp": [30.6],
    "fg": [6.3],
    "fga": [10.1],
    "fg%": [.616],
    "2p": [6.2],
    "2pa": [10],
    "2p%": [.621],
    "3p": [0],
    "3pa": [.1],
    "3p%": [.250],
    "ft": [3.7],
    "fta": [5.1],
    "ft%": [.720],
    "orb": [3.3],
    "drb": [5.4],
    "trb": [8.6],
    "ast": [1.1],
    "stl": [1.2],
    "blk": [2.7],
    "tov": [2.0],
    "pf": [2.7],
    "pts": [16.2],
    "sos": [7.22],
    "position": "forward",
    "height": "6-9",
    "handedness": "right"}
)

# 17.70
pipe.predict(okongwu)

haliburton = pd.DataFrame({
    'college_country': "Iowa State",
    "g": [57],
    "gs": [56],
    "mp": [34.6],
    "fg": [3.7],
    "fga": [7.2],
    "fg%": [.509],
    "2p": [1.9],
    "2pa": [3.1],
    "2p%": [.621],
    "3p": [1.8],
    "3pa": [4.2],
    "3p%": [.426],
    "ft": [1],
    "fta": [1.2],
    "ft%": [.775],
    "orb": [1],
    "drb": [3.4],
    "trb": [4.4],
    "ast": [4.7],
    "stl": [1.9],
    "blk": [.8],
    "tov": [1.6],
    "pf": [1.3],
    "pts": [10.1],
    "sos": [9.68],
    "position": "guard",
    "height": "6-5",
    "handedness": "right"}
)

# 9.53
pipe.predict(haliburton)

maxey = pd.DataFrame({
    'college_country': "Kentucky",
    "g": [31],
    "gs": [28],
    "mp": [34.5],
    "fg": [4.8],
    "fga": [11.3],
    "fg%": [.427],
    "2p": [3.8],
    "2pa": [7.7],
    "2p%": [.492],
    "3p": [1.1],
    "3pa": [3.6],
    "3p%": [.292],
    "ft": [3.2],
    "fta": [3.9],
    "ft%": [.833],
    "orb": [.4],
    "drb": [3.9],
    "trb": [4.3],
    "ast": [3.2],
    "stl": [.9],
    "blk": [.4],
    "tov": [2.2],
    "pf": [2.4],
    "pts": [14],
    "sos": [7.46],
    "position": "guard",
    "height": "6-3",
    "handedness": "right"}
)

# 11.31
pipe.predict(maxey)

cole = pd.DataFrame({
    'college_country': "North Carolina",
    "g": [22],
    "gs": [20],
    "mp": [34.9],
    "fg": [6],
    "fga": [15.7],
    "fg%": [.380],
    "2p": [3.7],
    "2pa": [9.3],
    "2p%": [.402],
    "3p": [2.2],
    "3pa": [6.4],
    "3p%": [.348],
    "ft": [4.4],
    "fta": [5.8],
    "ft%": [.750],
    "orb": [.5],
    "drb": [5.2],
    "trb": [5.7],
    "ast": [4],
    "stl": [1.3],
    "blk": [.3],
    "tov": [3.5],
    "pf": [2.9],
    "pts": [18.5],
    "sos": [10.05],
    "position": "guard",
    "height": "6-3",
    "handedness": "right"}
)

# 10.54
pipe.predict(cole)
# Highest in order: 1) wiseman 2) okongwu 3) edwards 4) topping 5) maxey
# 6) cole 7) haliburton 8) bey
