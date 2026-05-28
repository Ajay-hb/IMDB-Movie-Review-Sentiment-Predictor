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
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_LENGTH))
    model.add(LSTM(rnn_units))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource
def load_sentiment_model(path: str = MODEL_PATH):
    # Try loading existing saved model
    if os.path.exists(path):
        try:
            model = load_model(path)
            return model
        except Exception:
            pass  # Fall through to retrain if loading fails

    # No valid model found — train from scratch
    st.info("No pre-trained model found. Training now (~2–3 minutes)...")

    (X_train, y_train), _ = imdb.load_data(num_words=VOCAB_SIZE)
    X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)

    model = build_default_model(VOCAB_SIZE)

    progress_bar = st.progress(0, text="Training in progress...")
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=0)
        progress_bar.progress(
            int((epoch + 1) / EPOCHS * 100),
            text=f"Training epoch {epoch + 1}/{EPOCHS}..."
        )

    progress_bar.empty()
    model.save(path)
    st.success("Model trained and saved successfully!")
    return model


def main():
    st.set_page_config(page_title="IMDB Sentiment Predictor", layout="wide")
    st.title("🎬 IMDB Movie Review Sentiment Predictor")
    st.markdown(
        "Enter a movie review below and the app will predict whether it is **positive** or **negative**."
    )

    try:
        model = load_sentiment_model()
    except Exception as exc:
        st.error(f"Failed to load or train model: {exc}")
        return

    word_index = load_word_index()

    review_text = st.text_area("Enter your movie review:", height=180)

    if st.button("Predict Sentiment"):
        if not review_text.strip():
            st.warning("Please enter a review before clicking Predict Sentiment.")
        else:
            with st.spinner("Analysing..."):
                input_sequence = preprocess_review(review_text, word_index, MAX_LENGTH, VOCAB_SIZE)
                prediction = float(model.predict(input_sequence, verbose=0)[0][0])

            sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
            confidence = prediction if prediction >= 0.5 else 1 - prediction

            st.write("### Prediction")
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", sentiment)
            col2.metric("Confidence", f"{confidence:.1%}")

            st.progress(prediction, text=f"Positive probability: {prediction:.4f}")

    st.markdown("---")
    st.write("### Notes")
    st.write(
        "- The model is an LSTM trained on the IMDB dataset (25,000 reviews).\n"
        "- If no pre-trained `.h5` file is present, the model trains automatically on first launch.\n"
        "- Input text is preprocessed using the IMDB word index and padded to 200 tokens."
    )


if __name__ == "__main__":
    main()
