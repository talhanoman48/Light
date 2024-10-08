# Import necessary libraries
import json  # For working with JSON files
import nltk  # For natural language processing tasks
from nltk.stem.snowball import SnowballStemmer  # For stemming words (reducing words to their root form)
import numpy as np  # For numerical operations
import pickle  # For serializing and deserializing Python objects
import streamlit as st  # For displaying content in a Streamlit app
import os  # For interacting with the operating system (file paths, etc.)

# Initialize the SnowballStemmer for the English language
stemmer = SnowballStemmer('english')

# Get the current file's directory (useful for accessing files relative to the script's location)
here = os.path.dirname(os.path.abspath(__file__))

def load_json(name='intents.json'):
    """
    Load data from a JSON file.
    
    Parameters:
    - name (str): The name of the JSON file to load (default is 'intents.json').
    
    Returns:
    - data (dict): The loaded JSON data as a dictionary.
    """
    open_json = open('Databases/' + name)  # Open the specified JSON file from the 'Databases' folder
    data = json.load(open_json)  # Load the JSON data into a Python dictionary
    return data  # Return the data

def update_json(x, name):
    """
    Update a JSON file by adding a new intent to the 'intents' list.
    
    Parameters:
    - x (dict): The new intent to be added.
    - name (str): The name of the JSON file to update.
    """
    # Open the existing JSON file and load the data
    with open('Databases/' + name, 'r') as f:
        data = json.load(f)
    
    # Append the new intent (x) to the 'intents' list
    with open('Databases/' + name, 'w') as f:
        data['intents'].append(x)
        json.dump(data, f)  # Write the updated data back to the JSON file

def update_old_databases(name):
    """
    Update an old JSON database format by adding new fields to each intent.
    
    Parameters:
    - name (str): The name of the JSON file to update.
    
    The new fields added are 'context_set', 'chapter', 'equation', and 'image', which were not present in the old format.
    """
    # Define a new structure for the JSON data with additional fields
    y = {"intents": []}
    
    # Open the old database file and load the data
    with open('Datasets/' + name, 'r') as f:
        data = json.load(f)
    
    # Iterate over the intents and update each with the new structure
    for intent in data['intents']:
        x = {
            "tag": intent['tag'],  # Original intent tag
            "patterns": intent['patterns'],  # Original patterns
            "responses": intent['responses'],  # Original responses
            "context_set": "",  # New field, initially empty
            "chapter": "",  # New field, initially empty
            "equation": "",  # New field, initially empty
            "image": ""  # New field, initially empty
        }
        y['intents'].append(x)  # Add the updated intent to the new data
    
    # Write the updated data back to the file
    with open('Datasets/' + name, 'w') as f:
        json.dump(y, f)

def generate_dictionaries(file_name, subject_name):
    """
    Generate and save vocabularies and other related data from the intents in a JSON file.
    
    Parameters:
    - file_name (str): The name of the JSON file containing the intents.
    - subject_name (str): The subject name used to save the dictionaries as a pickle file.
    
    Returns:
    - vocab (list): List of unique stemmed words from the patterns.
    - dictionary (list): List of all stemmed words from the patterns.
    - labels_vocab (list): List of unique intent tags (labels).
    - classes (list): List of intent tags.
    - responses (list): List of all responses.
    - chapters (list): List of unique chapter names from the intents.
    """
    try:
        # Try loading from an existing pickle file
        with open(f'{subject_name}dictionaries.pickle', 'rb') as f:
            vocab = pickle.load(f)
            dictionary = pickle.load(f)
            labels_vocab = pickle.load(f)
            classes = pickle.load(f)
            responses = pickle.load(f)
            chapters = pickle.load(f)
        st.write("Loaded from pickle")
        return vocab, dictionary, labels_vocab, classes, responses, chapters
    except:
        # If pickle file doesn't exist, load data from JSON file and create dictionaries
        data = load_json(file_name)
        dictionary = []  # All stemmed words from patterns
        labels_vocab = []  # Unique intent tags
        classes = []  # List of all intent tags (same as labels_vocab)
        responses = []  # List of all responses
        chapter = []  # List of chapter names (if available in the intents)

        # Process each intent in the JSON data
        for intent in data['intents']:
            # Process each pattern in the intent and stem each word
            for pattern in intent['patterns']:
                for word in nltk.word_tokenize(stemmer.stem(pattern)):
                    dictionary.append(word)
            labels_vocab.append(intent['tag'])
            classes.append(intent['tag'])
            responses.extend(intent['responses'])
            if intent['chapter'] != "":
                chapter.append(intent['chapter'])

        # Remove duplicates and sort the vocab and chapters
        vocab = list(sorted(set(dictionary)))
        chapters = list(sorted(set(chapter)))

        # Save the generated data into a pickle file
        with open(f"{subject_name}dictionaries.pickle", "wb") as f:
            pickle.dump(vocab, f)
            pickle.dump(dictionary, f)
            pickle.dump(labels_vocab, f)
            pickle.dump(classes, f)
            pickle.dump(responses, f)
            pickle.dump(chapters, f)

        return vocab, dictionary, labels_vocab, classes, responses, chapters

def generate_data(file_name, vocab, labels_vocab):
    """
    Generate training data for patterns and labels from the intents.
    
    Parameters:
    - file_name (str): The name of the JSON file containing the intents.
    - vocab (list): The vocabulary generated from the intents' patterns.
    - labels_vocab (list): The list of intent tags.
    
    Returns:
    - patterns (np.array): Bag of words representation of the patterns.
    - labels (np.array): One-hot encoded representation of the intent tags.
    """
    try:
        # Try loading pre-processed data from saved numpy files
        patterns = np.load('/Patterns/patterns' + file_name, dtype=int)
        labels = np.load('/Labels/labels' + file_name, dtype=int)
        return patterns, labels
    except:
        # If pre-processed data doesn't exist, generate the data from JSON
        data = load_json(file_name)
        patterns = []
        labels = []

        # Process each intent and create the bag of words representation for patterns and labels
        for intent in data['intents']:
            for pattern in intent['patterns']:
                patterns.append(bag_of_words(pattern, vocab))
                labels.append(bag_of_words(intent['tag'], labels_vocab, label_process=True))

        # Convert patterns and labels into numpy arrays and save them
        patterns = np.squeeze(np.asarray(patterns))
        labels = np.squeeze(np.asarray(labels))
        np.save('Patterns/patterns' + file_name, patterns)
        np.save('Labels/labels' + file_name, labels)

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
