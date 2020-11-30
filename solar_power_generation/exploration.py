import pandas as pd

df = pd.read_csv("data/Plant_1_Generation_Data.csv")
df_1 = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")

df.head()
df_1.head()
df.info()
df_1.info()

df.columns = map(str.lower, df.columns)
df.describe()
len(df[df["daily_yield"] == 0])

df["plant_id"].value_counts()
df["date_time"].value_counts()
df["date_time"] = df["date_time"] + ":00"


df_1.columns = map(str.lower, df_1.columns)

df_1["date_time"].value_counts()

df_dates = pd.DataFrame(df_1.date_time.str.split('-').tolist(),index=df_1.ambient_temperature)

df_dates.head(1)
df_2 = df_dates.reset_index([0,'ambient_temperature'])

df_2.head(1)

### Possible issue with pandas - not returning what it should.
df_2[2].str.split(" ")[0]
