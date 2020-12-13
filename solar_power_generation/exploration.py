import pandas as pd

df = pd.read_csv("data/Plant_1_Generation_Data.csv")
df_1 = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")

df.head()
df_1.head()
df.info()
df_1.info()

df.columns = map(str.lower, df.columns)
df_1.columns = map(str.lower, df_1.columns)
df.describe().T
df_1.describe().T

len(df[df["daily_yield"] == 0])
df["plant_id"].value_counts()

df["date_time"].value_counts()
df_1["date_time"].value_counts()

df["date_time"] = df["date_time"] + ":00"

df_column_changes = df.date_time.str.split("-",expand=True)
df_column_changes.head(1)
df_column_changes["more_changes"] = df_column_changes[2]
df_column_changes_1 = df_column_changes.more_changes.str.split(" ", expand=True)
df_column_changes_1.head(1)

df_column_changes_1["time"] = df_column_changes_1[0] + "-" + df_column_changes[1] + "-" + df_column_changes[0] + " " + df_column_changes_1[1]

df_column_changes_1.head(1)

df["date_time"] = df_column_changes_1["time"]

df.info()
df.head(1)
df_1.head(1)
