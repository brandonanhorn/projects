import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

df = pd.read_csv("clean_dog_data.csv")

df.head(1)

df_image = []
for i in tqdm(range(df.shape[0])):
    img = image.load_img(df['images'][i], target_size=(128,128,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    df_image.append(img)

X = np.array(df_image)
y = df["name_of_dog"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

X[0].dtype

# Old Model - poor performing
#model = Sequential()
#model.add(Flatten())
#model.add(Dense(960, input_shape=X_train[0], activation='relu'))
#model.add(Dense(480, activation='relu'))
#model.add(Dense(360, activation='relu'))
#model.add(Dense(240, activation='elu'))
#model.add(Dense(120, activation='softmax'))

## better performing model
model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=X_train[0]))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(120, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    epochs=5,
                    verbose=1)


## future testing
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img(df_test['images'][i], target_size=(128,128,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)

test = np.array(test_image)

prediction = model.predict_classes(test)
