import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('games_updated.csv')
pd.set_option('display.max_columns', None)

df.head(2)
del df["Unnamed: 0"]

target = 'winner'
y = df[target]
X = df.drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    (['turns'], StandardScaler()),
    ('opening_eco', LabelBinarizer()),
    ('opening_name', LabelBinarizer()),
    ('firstplay', LabelBinarizer()),
    ('secondplay', LabelBinarizer()),
    ('thirdplay', LabelBinarizer()),
    ('fourthplay', LabelBinarizer()),
    ('fifthplay', LabelBinarizer()),
    ('sixthplay', LabelBinarizer()),
    ('seventhplay', LabelBinarizer()),
    ('eighthplay', LabelBinarizer()),
    ('ninthplay', LabelBinarizer()),
    ('tenthplay', LabelBinarizer()),
#    ('eleventhplay', LabelBinarizer()),
#    ('twelvthplay', LabelBinarizer()),
#    ('thirteenthplay', LabelBinarizer()),
#    ('fourteenthplay', LabelBinarizer()),
#    ('fifteenthplay', LabelBinarizer()),
#    ('sixteenthplay', LabelBinarizer()),
#    ('seventhplay', LabelBinarizer()),
#    ('eigteenthplay', LabelBinarizer()),
#    ('nineteenthplay', LabelBinarizer()),
#    ('twentythplay', LabelBinarizer())
    ],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = LogisticRegression().fit(Z_train,y_train)
model.score(Z_train,y_train)
model.score(Z_test,y_test)

model = RandomForestClassifier().fit(Z_train,y_train)
model.score(Z_train,y_train)
model.score(Z_test,y_test)

model = DecisionTreeClassifier().fit(Z_train,y_train)
model.score(Z_train,y_train)
model.score(Z_test,y_test)

## Overfitting with the trees
