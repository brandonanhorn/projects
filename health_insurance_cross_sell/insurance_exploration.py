import pandas as pd

df = pd.read_csv("data/train.csv")
pd.set_option("max.columns", None)


df.head(1)
df.info()

df["Response"].mean()

## super clean data set onto modeling
