# Import necessary libraries
import numpy as np  # For handling numerical operations
import streamlit as st  # For displaying output in a Streamlit app

# Initialize an empty dictionary to store word embeddings and a list to store the GloVe vocabulary
embedding = {}
glove_vocab = []

def get_glove_vocab(embedding_dim, vocab_size, glove_vocab_size=6):
    """
    Load and display the vocabulary from the GloVe embeddings file.
    
    Parameters:
    - embedding_dim (int): The dimensionality of the word vectors in the GloVe file (e.g., 100 for 100D embeddings).
    - vocab_size (int): The number of words in the vocabulary (currently unused in the function, placeholder for future use).
    - glove_vocab_size (int): The size of the GloVe corpus used (default is 6B, i.e., 6 billion tokens).

    This function reads the GloVe file, extracts the words (vocabulary), and stores them in the `glove_vocab` list.
    It then joins the words into a single string and displays the vocabulary using Streamlit's `st.write`.
    """
    with open(f"glove.{glove_vocab_size}B.{embedding_dim}d.txt", encoding='utf-8') as f:
        for line in f:
            # Split each line into word and its corresponding vector values
            values = line.split()
            # Append the word (first element) to the GloVe vocabulary list
            glove_vocab.append(values[0])
    
    # Join the words in the vocabulary into a single string
    vocab = " ".join(w for w in glove_vocab)
    
    # Display the vocabulary in the Streamlit app
    st.write(vocab)

def load_embed(embedding_dim, vocab_size, glove_vocab_size=6):
    """
    Load the GloVe word embeddings into a dictionary.

    Parameters:
    - embedding_dim (int): The dimensionality of the word vectors in the GloVe file (e.g., 100 for 100D embeddings).
    - vocab_size (int): The number of words in the vocabulary (currently unused in the function, placeholder for future use).
    - glove_vocab_size (int): The size of the GloVe corpus used (default is 6B, i.e., 6 billion tokens).

    This function reads the GloVe file and stores each word and its corresponding vector in the `embedding` dictionary,
    where the key is the word, and the value is its vector (as a numpy array).
    """
    try:
        # Open the GloVe file and read it line by line
        with open(f"glove.{glove_vocab_size}B.{embedding_dim}d.txt", encoding='utf-8') as f:
            for line in f:
                # Split each line into the word and its corresponding vector values
                values = line.split()
                # Store the word (first element) and its vector (rest of the elements as a numpy array) in the embedding dictionary
                embedding[values[0]] = np.asarray(values[1:], dtype='float32')
    except IOError:
        # Handle the case where the GloVe file cannot be opened or found
        pass

    # Create an embedding matrix initialized to zeros with shape (vocab_size, embedding_dim)
    embedding_matrix = np.zeros((len(embedding.keys()), embedding_dim))
    return embedding_matrix

# Load the GloVe embeddings with a dimensionality of 100 and a vocabulary size of 263
load_embed(100, 263)
