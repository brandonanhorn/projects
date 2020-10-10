import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ibm_personal.csv")
df.head(2)

df.columns = [columns.lower() for columns in df.columns]
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
df.head(2)
df.info()

target = "performancerating"
X = df.drop(target, axis=1)
y = df[target]

plt.scatter(df["age"], df["department"]);

plt.plot(df["age"])
