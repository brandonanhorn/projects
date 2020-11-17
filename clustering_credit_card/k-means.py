%matplotlib inline
import pandas as pd

from sklearn.cluster import KMeans, k_means
from sklearn.metrics import silhouette_score

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_csv("data/CC General.csv")

df.head(1)
df.columns = map(str.lower, df.columns)

df.info()
df_x = df.drop("cust_id", axis=1)
df_x = df_x.fillna(0)

model = KMeans(n_clusters=5)
model.fit(df_x)
df["predict"] = model.labels_

df["predict"].value_counts()

plt.figure(figsize=(7,7));

colors = ["red", "green", "blue", "yellow", "black"]
df ["color"] = df["predict"].map(lambda p: colors[p])

ax = df.plot(
    kind = "scatter",
    x = "balance", y= "purchases",
    figsize = (10,8),
    c = df["color"]
)
