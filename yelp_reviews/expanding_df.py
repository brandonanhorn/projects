import pandas as pd

df = pd.read_csv("data/yelp.csv")

df.head(2)
df_2 = pd.DataFrame(df.text.str.split(" ").tolist(), index=df.review_id)
df_2.head(1)

df_2_reset_index = df_2.reset_index([0, "review_id"])

df = pd.merge(df, df_2_reset_index, on="review_id")

df.head(1)
del df["text"]

#trying to keep an average sentence length(about 20 words)
df = df.drop(df.columns[29:], axis=1)

pd.set_option("max.columns", None)

df.head(1)

df = df.rename(columns={0:"first_word", 1:"second_word", 2:"third_word", 3:"fourth_word",
                    4:"fifth_word", 5:"sixth_word", 6:"seventh_word", 7:"eighth_word",
                    8:"ninth_word", 9:"tenth_word", 10:"elevnth_word", 11:"twelfth_word",
                    12:"thirteenth_word", 13:"fourteenth_word", 14:"fifteenth_word",
                    15:"sixteenth_word", 16:"seventeenth_word", 17:"eighteenth_word",
                    18:"nineteenth_word", 19:"twentieth_word" })

df.head()
df.info()
df = df.fillna("n/a")

df.info()

df.to_csv("data/changed_data.csv")
