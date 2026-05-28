import os
import re
import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

MODEL_PATH = "lstm_sentiment_model.h5"
MAX_LENGTH = 200
VOCAB_SIZE = 10000

@st.cache_data
def load_word_index():
    raw_word_index = imdb.get_word_index()
    word_index = {word: index + 3 for word, index in raw_word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    return word_index

def preprocess_review(review: str, word_index: dict, maxlen: int, vocab_size: int):
    review = review.lower()
    review = re.sub(r"[^a-z0-9\s']", "", review)
    tokens = review.split()
    sequence = [1]
    for word in tokens:
        idx = word_index.get(word, 2)
        if idx >= vocab_size:
            idx = 2
        sequence.append(idx)
    return pad_sequences([sequence], maxlen=maxlen)

def build_default_model(vocab_size: int, embedding_dim: int = 32, rnn_units: int = 32):
    model = Sequential()
    model.add(Embedding(input_dim=
