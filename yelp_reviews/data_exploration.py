import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/yelp.csv")
pd.set_option("max.columns", None)

df.head(1)

plt.scatter(df["stars"], df["useful"]);

plt.scatter(df["stars"], df["funny"]);
