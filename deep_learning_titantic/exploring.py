import pandas as pd

df = pd.read_csv("data/train.csv")

df.head(3)

df.info()

df.describe().T

surnames = []

for name in df["Name"]:
    print(name.split(",")[1].split(".")[0])

a_list = [name.split(",")[1].split(".")[0].strip() for name in df["Name"]]


df["surnames"] = a_list

df["surnames"].value_counts()
mr = []

df.to_csv("clean_data.csv")
#def mr_count(s):
#    if "Mr." in s:
#        return 1
#    else:
#        return 0
#
#df['Mr'] = df["Name"].map(mr_count)
#
#pd.concat([df, pd.get_dummies(df["Embarked"], prefix="Embarked")], axis=1)
#df.head(3)
#
#((df["Fare"] > 30) & (df["Sex"] == "male")).sum()
#
#
#a_female = [int(each == "female") for each in df["Sex"]]
#a_female
