import nltk
from nltk.stem import PorterStemmer
import numpy as np

stemmer = PorterStemmer()


#splits sentence into tokenized words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


#stems each word in the bag
def stem(word):
    return stemmer.stem(word.lower())


#creates an numpy array with 0s and 1s to accurately predict which message the user inputted
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[i] = 1.0
    return bag

