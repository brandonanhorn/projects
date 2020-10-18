 import pandas as pd

df = pd.read_csv("data/beach-water-quality-automated-sensors-1.csv")

df.head()
df.info()
del df["Transducer Depth"]

df = df.dropna()

x = []
y = []

for i in df["Measurement Timestamp"]:
    x_1 = i.split(" ")[0]
    x.append(x_1)
    y_1 = i.split(" ")[1]
    y.append(y_1)

df["rework time"] = x
df["time of day"] = y

month = []
day = []
year = []

for i in df["rework time"]:
    x = i.split("/")[0]
    month.append(x)
    y = i.split("/")[1]
    day.append(y)
    z = i.split("/")[2].split(" ")[0]
    year.append(z)

df["month"] = month
df["day"] = day
df["year"] = year

df["day"].value_counts()

df.head(1)

del df["rework time"]
del df["Measurement Timestamp"]
del df["Measurement Timestamp Label"]
del df["Measurement ID"]

df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = map(str.lower, df.columns)

df.info()

df.to_csv("data/clean_data.csv")
