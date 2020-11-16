import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random
import json
import pickle
import time

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

import logging, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

with open("data.json", encoding="utf-8") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["data"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='elu'))
model.add(Dropout(0.1))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

try:
    model.load("model.h5")

except:
    hist = model.fit(training, output, epochs=19, batch_size=1, verbose=1)
    model.save('model.h5', hist)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for sent in s_words:
        for i, w in enumerate(words):
            if w == sent:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with Brandon")
    time.sleep(.3)
    print("Type 'quit' at anytime to close the chatbot ")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Thank you for the talk. All the best to you.")
            time.sleep(.3)
            print("Please reach out to me at brandon.anhorn@gmail.com if you'd like to talk to the person behind the bot.")
            break

        results = model.predict([[bag_of_words(inp, words)]])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for ta in data["data"]:
            if ta["tag"] == tag:
                responses = ta["responses"]

        print(random.choice(responses))

chat()
