import json
import random
from nltk_utils import tokenize, stem, bag_of_words
import pickle
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

ignored_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words_in_pattern = [stem(w) for w in tokenize(pattern) if w not in ignored_words]
        all_words.extend(words_in_pattern)
        xy.append((words_in_pattern, tag))

all_words = sorted(set(all_words))
tags = sorted(set(tags))

pickle.dump(all_words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))

# Build training data (x and y axis)
training = []
for document in xy:
    bag = []
    pattern_words = document[0]
    pattern_words = [stem(w) for w in pattern_words]
    for w in all_words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = [0] * len(tags)
    output_row[tags.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
x_train = list(training[:, 0])
y_train = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Convert x and y axis into np arrays before fitting the model
x_train = np.array(x_train)
y_train = np.array(y_train)

hist = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')