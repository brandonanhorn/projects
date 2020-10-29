import pandas as pd

df = pd.read_csv("data/train.csv")
pd.set_option("max_columns", None)

df.head(2)

df.info()

df["bathrooms"] = df["bathrooms"].fillna(0)
df["review_scores_rating"] = df["review_scores_rating"].fillna(0)
df["bedrooms"] = df["bedrooms"].fillna(1)
df["beds"] = df["beds"].fillna(0)
df["zipcode"] = df["zipcode"].fillna("11111")
df["neighbourhood"] = df["neighbourhood"].fillna("unknown")
df["host_has_profile_pic"] = df["host_has_profile_pic"].fillna("f")
df["host_identity_verified"] = df["host_identity_verified"].fillna("f")
df["host_since"] = df["host_since"].fillna("new")
del df["host_response_rate"]
df.info()

df_amenities = pd.DataFrame(df.amenities.str.split(',').tolist(),index=df.id)
df_amenities_reset = df_amenities.reset_index([0, "id"])

df = pd.merge(df, df_amenities_reset, on="id")

df[0] = df[0].str.replace("{", "")
df[0] = df[0].str.replace('"', "")
df[1] = df[1].str.replace('"', "")
df[2] = df[2].str.replace('"', "")
df[3] = df[3].str.replace('"', "")
df[4] = df[4].str.replace('"', "")
df[5] = df[5].str.replace('"', "")
df[6] = df[6].str.replace('"', "")

df.rename(columns={0:'amenity_1', 1:'amenity_2', 2:'amenity_3', 3:'amenity_4',
                    4:'amenity_5', 5:'amenity_6', 6:'amenity_7'},inplace=True)

df.drop(df.columns[32:], axis=1, inplace=True)
del df["id"]
del df["amenities"]
del df["description"]

df["amenity_2"] = df["amenity_2"].fillna("n/a")
df["amenity_3"] = df["amenity_3"].fillna("n/a")
df["amenity_4"] = df["amenity_4"].fillna("n/a")
df["amenity_5"] = df["amenity_5"].fillna("n/a")
df["amenity_6"] = df["amenity_6"].fillna("n/a")
df["amenity_7"] = df["amenity_7"].fillna("n/a")

df["amenity_2"] = df["amenity_2"].astype("str")
df["amenity_3"] = df["amenity_3"].astype("str")
df["amenity_4"] = df["amenity_4"].astype("str")
df["amenity_5"] = df["amenity_5"].astype("str")
df["amenity_6"] = df["amenity_6"].astype("str")
df["amenity_7"] = df["amenity_7"].astype("str")

df.to_csv("reworked_data.csv")

df.head(2)



mapper = DataFrameMapper([
    ("property_type", LabelBinarizer()),
    ("room_type", LabelBinarizer()),
    (["accommodates"], StandardScaler()),
    (["bathrooms"], StandardScaler()),
    ("bed_type", LabelBinarizer()),
    ("cancellation_policy", LabelBinarizer()),
    ("cleaning_fee", LabelBinarizer()),
    ("city", LabelBinarizer()),
    ("host_has_profile_pic", LabelBinarizer()),
    ("host_identity_verified", LabelBinarizer()),
    ("host_since", LabelBinarizer()),
    ("instant_bookable", LabelBinarizer()),
    ("neighbourhood", LabelBinarizer()),
    (["number_of_reviews"], StandardScaler()),
    (["review_scores_rating"], StandardScaler()),
    ("zipcode", LabelBinarizer()),
    (["bedrooms"], StandardScaler()),
    (["beds"], StandardScaler()),
    ("amenity_1", LabelBinarizer()),
    ("amenity_2", LabelBinarizer()),
    ("amenity_3", LabelBinarizer()),
    ("amenity_4", LabelBinarizer()),
    ("amenity_5", LabelBinarizer()),
    ("amenity_6", LabelBinarizer()),
    ("amenity_7", LabelBinarizer())],df_out=True)
