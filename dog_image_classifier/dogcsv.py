import os
import pandas as pd

# all of the images
base_dir = 'images/'
rootdir = base_dir+'stanford-dogs-dataset/images/Images/'

all_images =  []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        all_images.append(os.path.join(subdir, file))

df = pd.DataFrame()
df['images'] = all_images
df.to_csv('dogs.csv')

pd.read_csv('dogs.csv')

#all of the annotations
rootdir_1 = base_dir+'stanford-dogs-dataset/annotations/Annotation/'

all_annotations =  []
for subdir, dirs, files in os.walk(rootdir_1):
    for file in files:
        all_annotations.append(os.path.join(subdir, file))

df_1 = pd.DataFrame()
df_1['images'] = all_annotations
df_1.to_csv('annotations.csv')

pd.read_csv('annotations.csv')


df['label'] = df['images']

del df["label"][0]
del df["images"][0]

df

df = df.drop(df.index[0])

c = [i.split("/")[4] for i in df["label"]]
df["name_of_dog"] = [ii.split("-")[1] for ii in c]


df = df.drop("label", axis=1)
df ["name_of_dog"] = df["name_of_dog"].str.lower()
df["name_of_dog"] = df["name_of_dog"].str.replace("_"," ")
df.to_csv("clean_dog_data.csv")
