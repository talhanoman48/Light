import json
import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pickle
import streamlit as st
import os

stemmer = SnowballStemmer('english')
here = os.path.dirname(os.path.abspath(__file__))


def load_json(name = 'intents.json'):
    open_json = open('Databases/' + name)
    data = json.load(open_json)
    return data


def update_json(x, name):
    with open('Databases/' + name, 'r') as f:
        data = json.load(f)
    with open('Databases/' + name, 'w') as f:
        data['intents'].append(x)
        json.dump(data, f)


def update_old_databases(name):
    y = {
    "intents":[]
    }
    with open('Datasets/' + name, 'r') as f:
        data = json.load(f)
    for intent in data['intents']:
        x = {
            "tag": intent['tag'],
            "patterns": intent['patterns'],
            "responses": intent['responses'],
            "context_set": "",
            "chapter": "",
            "equation":"",
            "image": ""
            }
        y['intents'].append(x)
    with open('Datasets/' + name, 'w') as f:
        json.dump(y, f)


def generate_dictionaries(file_name, subject_name):
    try:
        with open(f'{subject_name}dictionaries.pickle', 'rb') as f:
            #vocab, dictionary, labels_vocab, classes, responses, chapters = pickel.load(f)
            vocab = pickle.load(vocab, f)
            dictionary = pickle.load(dictionary, f)
            labels_vocab = pickle.load(labels_vocab, f)
            classes = pickle.load(classes, f)
            responses = pickle.load(responses, f)
            chapters = pickle.load(chapters, f)
        return vocab, dictionary, labels_vocab, classes, responses, chapters
        st.write("loaded from pickle")
    except:
        data = load_json(file_name)
        dictionary = []
        labels_vocab = []
        classes = []
        responses = []
        chapter = []
        for intent in data['intents']:
            for pattern in intent['patterns']:
                for word in nltk.word_tokenize(stemmer.stem(pattern)):
                    dictionary.append(word)
            labels_vocab.append(intent['tag'])
            classes.append(intent['tag'])
            responses.extend(intent['responses'])
            if intent['chapter'] != "":
                chapter.append(intent['chapter'])
        vocab = list(sorted(set(dictionary)))
        chapters = list(sorted(set(chapter)))
        with open(f"{subject_name}dictionaries.pickle", "wb") as f:
            #pickle.dump(vocab, dictionary, labels_vocab, classes, responses, chapters, f)
            pickle.dump(vocab, f)
            pickle.dump(dictionary, f)
            pickle.dump(labels_vocab, f)
            pickle.dump(classes, f)
            pickle.dump(responses, f)
            pickle.dump(chapters, f)
        return vocab, dictionary, labels_vocab, classes, responses, chapters


def generate_data(file_name, vocab, labels_vocab):
    try:
        patterns = np.load('/Patterns/patterns'+file_name, dtype = int)
        labels = np.load('/Labels/labels'+ file_name, dtype = int)
        return patterns, labels
    except:
        data = load_json(file_name)
        patterns = []
        labels = []
        for intent in data['intents']:
            for pattern in intent['patterns']:
                patterns.append(bag_of_words(pattern, vocab))
                labels.append(bag_of_words(intent['tag'], labels_vocab,
                                            label_process = True))
        patterns = np.squeeze(np.asarray(patterns))
        labels = np.squeeze(np.asarray(labels))
        np.save('Patterns/patterns'+file_name, patterns)
        np.save('Labels/labels'+ file_name, labels)
        return patterns, labels


def bag_of_words(input, vocab, label_process = False):
    '''
    Returns the input in the form of a numpy array of zeros and ones

    Parameters:
    input - string that the user inputs
    vocab - sorted list of the vocabulary words
    '''
    if label_process == False:
        input_words = nltk.word_tokenize(stemmer.stem(input))
        vocab_dictionary = dict((k,i) for i, k in enumerate(vocab))
        intersection = set(vocab_dictionary).intersection(set(input_words))
        indices_input_vocab = [vocab_dictionary[x] for x in intersection]
        vocab_np = np.asarray(vocab).reshape(len(vocab),1)
        nn_input = np.zeros(vocab_np.shape)
        for x in indices_input_vocab:
            nn_input[x] = 1
        return nn_input
    else:
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
        else:
            st.write(input)
