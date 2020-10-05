import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv("data/nba_college_players.csv")

df.head()

df["height"] = df['height'].str.replace('"', "")

df.info()

df.describe().T

## seems to favor big guys
df.sort_values(by="per", ascending=False)

## seems to be the best indicator, data for vorp may need to be divided by # of years played
df.sort_values(by="vorp", ascending=False)

df["per"].mean()

df["per"].plot.hist();
