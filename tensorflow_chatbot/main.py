import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import random
import pickle

words=[]
classes = []
docs = []
ignore_words = ['?', '!']
with open("data.json") as file:
    data = json.load(file)


for dat in data["data"]:
    for pattern in dat["patterns"]:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        docs.append((w, dat['tag']))

        if dat['tag'] not in classes:
            classes.append(dat['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in docs:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

x_train = list(training[:,0])
y_train = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

hist = model.fit(np.array(x_train), np.array(y_train), epochs=40, batch_size=5, verbose=1)
model.save('model_1.h5', hist)

print("Everything worked.")
## Appears to be overfit
