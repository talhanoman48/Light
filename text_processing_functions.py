import json
from nltk.corpus import stopwords
from collections import Counter
import pickle
import numpy as np
import string
import time
import streamlit as st

from tensorflow.python.eager.context import execution_mode



#stop = set(stopwords.words('english'))


def load_data(file_name):
    '''
    Loads data from json into an object so that it can be manipulated

    Parameters:
    file_name(type: string): The name of the json file
    '''
    file_name = file_name + ".json"
    with open("Datasets/"+file_name) as f:
        data = json.load(f)
    return data


def remove_punct(text):
    '''
    Removes punctuation from provided string
    '''
    table = str.maketrans("","",string.punctuation)
    return text.translate(table)


def remove_stopwords(text):
    '''
    Removes stopwords from provided string
    '''
    text = [word.lower() for word in text.split() if word not in stop]
    return " ".join(text)


def counter_word(text):
    '''
    Generates a collection containing the word as key and the number of times it appears in the data as value
    '''
    count = Counter()
    for i in text:
        for word in i.split():
            count[word] += 1
    return count


def gen_dictionaries(file_name):
    '''
    Generates the Dictionaries of:
    1. Patterns: the various inputs of the user
    2. Labels: The different categories that the user input can be classified into
    '''
    try:
        tic = time.time()
        with open("dictionaries/"+file_name+"dictionaries.pickle", "rb") as f:
            patterns = pickle.load(f)
            classes = pickle.load(f)
            patterns_counter = pickle.load(f)
            responses = pickle.load(f)
        toc = time.time()
        execution_time= toc - tic
        return patterns_counter, patterns, classes, responses, execution_time
    except:
        tic = time.time()
        patterns = []
        classes = []
        responses = []
        data = load_data(file_name)
        for intent in data['intents']:
            for pattern in intent['patterns']:
                text = remove_punct(pattern)
                #processed_text = remove_stopwords(text)
                patterns.append(text)
            classes.append(intent['tag'])
            for response in intent['responses']:
                responses.append(response)
        patterns_counter = counter_word(patterns)
        with open("dictionaries/"+file_name+"dictionaries.pickle", "wb") as f:
            pickle.dump(patterns, f)
            pickle.dump(classes, f)
            pickle.dump(patterns_counter, f)
            pickle.dump(responses, f)
        #classes = list(sorted(set(classes)))
        toc = time.time()
        execution_time= toc - tic
        return patterns_counter, patterns, classes, responses, execution_time


def bag_of_words(input, vocab):
    '''
    Returns the input string in the form of an array conatining zeros and ones.

    Parameters:
    input(type: string)
    vocab(type: list)
    '''
    vocab_dictionary = dict((k,i) for i, k in enumerate(vocab))
    input_words  = [input]
    if set(vocab_dictionary).intersection(set(input_words)):
        intersection = set(vocab_dictionary).intersection(set(input_words))
        indices = [vocab_dictionary[x] for x in intersection]
        vocab_np = np.asarray(vocab).reshape(len(vocab), 1)
        nn_outputs = np.zeros(vocab_np.shape)
        for x in indices:
            nn_outputs[x] = 1
        return nn_outputs

def generate_labels(classes, file_name):
    '''
    Generates labels for the input list

    Parameters:
    classes(type: list)
    file_name(type: str)
    '''
    data = load_data(file_name)
    labels = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            labels.append(bag_of_words(intent['tag'], classes))
    labels = np.squeeze(np.asarray(labels))
    np.save('Labels/labels'+ file_name, labels)
    return labels

def stemText(patterns, stemmer):
    '''
    Stemms the list of sentences and returns the processed list with stemmed words to
    reduce the size of the vocabulary.
    '''
    output = []
    for pattern in patterns:
        pattern = pattern.lower()
        output.append(" ".join([stemmer.lemmatize(i) for i in pattern.split()]))
    return output

def removeStopwords(patterns, stop_words):
    '''
    Removing stop words from list of strings
    '''
    filtered_sentences = []
    for pattern in patterns:
        filtered_sentences.append(" ".join([w for w in pattern.split() if not w.lower() in stop_words]))
    return filtered_sentences