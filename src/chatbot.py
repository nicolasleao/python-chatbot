import random
import json
import pickle
import numpy as np

from training.nltk_utils import tokenize, stem, bag_of_words

from tensorflow import keras
from keras.models import load_model

intents = json.loads(open('training/intents.json').read())
words = pickle.load(open('training/words.pkl', 'rb'))
tags = pickle.load(open('training/tags.pkl', 'rb'))
model = load_model('training/chatbot_model.h5')

def clear_input(sentence):
    sentence_words = tokenize(sentence)
    sentence_words = [stem(word) for word in sentence_words]
    return sentence_words

def classify_input(sentence):
    bow = bag_of_words(clear_input(sentence), words)
    bow = np.array([bow])
    tagged_input = model.predict(bow)[0]
    threshold = 0.3
    results = [[i, r] for i, r in enumerate(tagged_input) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': tags[r[0]], 'probability': str(r[1])})
    return return_list

def chat(sentence):
    classified = classify_input(sentence)
    tag = classified[0]['intent']
    probability = classified[0]['probability']
    print()
    print('Tipo de interação: ' + str(tag))
    print('Certeza: ' + str(probability))
    result = 'Desculpe, não entendi o que você disse.'
    for i in intents['intents']:
        if i['tag'] == tag:
          result = random.choice(i['responses'])
          break
    print('Resposta: ' + str(result))
    print()
        
while True:
    chat(input('You: '))