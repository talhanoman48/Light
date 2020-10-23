import numpy as np
import streamlit as st


embedding = {}
glove_vocab = []
def get_glove_vocab(embedding_dim, vocab_size, glove_vocab_size=6):
    with open(f"glove.{glove_vocab_size}B.{embedding_dim}d.txt", encoding='utf-8') as f:
            for line in f:
                values = line.split()
                glove_vocab.append(values[0])
    vocab = " ".join(w for w in glove_vocab)
    st.write(vocab)


def load_embed(embedding_dim, vocab_size, glove_vocab_size=6):
    try:
        with open(f"glove.{glove_vocab_size}B.{embedding_dim}d.txt", encoding='utf-8') as f:
            for line in f:
                values = line.split()
                embedding[values[0]] = np.asarray(values[1:], dtype = 'float32')
    except IOError:
        pass
    embedding_matrix = np.zeros((len(embedding.keys()), embedding_dim))
    return 

    

load_embed(100, 263)