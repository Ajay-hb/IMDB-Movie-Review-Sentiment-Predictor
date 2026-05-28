import os
import re
import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
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
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_LENGTH))
    model.add(LSTM(rnn_units))
    model.add(Dense(1, activation="sigmoid"))
    return model

@st.cache_resource
def load_sentiment_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Please place the saved Keras weights file in the same folder."
        )
    model = build_default_model(VOCAB_SIZE)
    model.load_weights(path)
    return model


def main():
    st.set_page_config(page_title="IMDB Sentiment Predictor", layout="wide")
    st.title("IMDB Movie Review Sentiment Predictor")
    st.markdown(
        "Enter a movie review below and the app will predict whether it is positive or negative using a prebuilt sentiment model."
    )



    global MAX_LENGTH
    MAX_LENGTH = 200
    VOCAB_SIZE = 10000

    try:
        model = load_sentiment_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    word_index = load_word_index()

    review_text = st.text_area("Enter your movie review:", height=180)
    if st.button("Predict sentiment"):
        if not review_text.strip():
            st.warning("Please enter a review before clicking Predict sentiment.")
        else:
            input_sequence = preprocess_review(review_text, word_index, MAX_LENGTH, VOCAB_SIZE)
            prediction = model.predict(input_sequence, verbose=0)[0][0]
            sentiment = "Positive" if prediction >= 0.5 else "Negative"
            st.write("### Prediction")
            st.write(f"**Review sentiment:** {sentiment}")
            st.write(f"**Positive probability:** {prediction:.4f}")

    st.markdown("---")
    st.write("### Notes")
    st.write(
        "- This app uses a prebuilt model and does not train in the browser."
        "\n- Save a compatible Keras model as `lstm_sentiment_model.h5` in the same folder as this script."
        "\n- The input text is preprocessed using the IMDB word index and padded to 200 tokens."
    )


if __name__ == "__main__":
    main()
