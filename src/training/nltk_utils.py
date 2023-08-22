import nltk
import numpy as np

# nltk.download('punkt')
stemmer = nltk.stem.SnowballStemmer("portuguese")

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    result = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            result[idx] = 1.0
    
    return result