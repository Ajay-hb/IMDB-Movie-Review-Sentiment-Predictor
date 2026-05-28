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

# ── Custom CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0a0a0f !important;
        color: #e8e6e0 !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { display: none; }
    .block-container {
        max-width: 820px !important;
        padding: 2rem 2rem 4rem !important;
        margin: 0 auto !important;
    }

    /* ── Typography ── */
    * { font-family: 'DM Sans', sans-serif !important; }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: 3.5rem 0 2.5rem;
        position: relative;
    }
    .hero-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #e8c36a;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: clamp(3.2rem, 8vw, 5.5rem);
        line-height: 0.95;
        color: #f0ede6;
        letter-spacing: 0.02em;
        margin: 0 0 1rem;
    }
    .hero-title span { color: #e8c36a; }
    .hero-subtitle {
        font-size: 1rem;
        color: #9c9a94;
        font-weight: 300;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .hero-divider {
        width: 48px;
        height: 2px;
        background: #e8c36a;
        margin: 1.8rem auto 0;
    }

    /* ── Input card ── */
    .input-card {
        background: #13131a;
        border: 1px solid #2a2a35;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0 1.5rem;
    }
    .input-label {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #9c9a94;
        margin-bottom: 0.75rem;
    }

    /* ── Textarea override ── */
    textarea {
        background: #0d0d14 !important;
        border: 1px solid #2a2a35 !important;
        border-radius: 10px !important;
        color: #e8e6e0 !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        padding: 1rem !important;
        resize: vertical !important;
        caret-color: #e8c36a !important;
    }
    textarea:focus {
        border-color: #e8c36a !important;
        box-shadow: 0 0 0 3px rgba(232,195,106,0.08) !important;
        outline: none !important;
    }

    /* ── Button ── */
    .stButton > button {
        width: 100% !important;
        background: #e8c36a !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        padding: 0.85rem 1.5rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin-top: 1rem !important;
    }
    .stButton > button:hover {
        background: #f0d080 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(232,195,106,0.25) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Result card ── */
    .result-card {
        background: #13131a;
        border: 1px solid #2a2a35;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .result-label {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #9c9a94;
        margin-bottom: 0.5rem;
    }
    .result-sentiment-positive {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 3.2rem;
        color: #6ecf8f;
        letter-spacing: 0.05em;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .result-sentiment-negative {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 3.2rem;
        color: #e8706a;
        letter-spacing: 0.05em;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .result-confidence {
        font-size: 0.9rem;
        color: #9c9a94;
        font-weight: 300;
    }
    .result-confidence strong { color: #e8e6e0; font-weight: 500; }

    /* ── Probability bar ── */
    .prob-bar-wrap {
        margin: 1.5rem 0 0;
        text-align: left;
    }
    .prob-bar-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9c9a94;
        margin-bottom: 0.4rem;
    }
    .prob-bar-track {
        height: 6px;
        background: #1e1e28;
        border-radius: 99px;
        overflow: hidden;
    }
    .prob-bar-fill-pos {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #4dba72, #6ecf8f);
        transition: width 0.6s ease;
    }
    .prob-bar-fill-neg {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #c94040, #e8706a);
        transition: width 0.6s ease;
    }

    /* ── Stats row ── */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .stat-box {
        flex: 1;
        background: #13131a;
        border: 1px solid #2a2a35;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .stat-value {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 1.8rem;
        color: #e8c36a;
        line-height: 1;
    }
    .stat-key {
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #9c9a94;
        margin-top: 0.3rem;
    }

    /* ── Training info ── */
    .stAlert {
        background: #13131a !important;
        border: 1px solid #2a2a35 !important;
        border-radius: 12px !important;
        color: #9c9a94 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: #e8c36a !important;
        border-radius: 99px !important;
    }
    .stProgress > div {
        background: #1e1e28 !important;
        border-radius: 99px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #e8c36a !important; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 3rem 0 1rem;
        font-size: 0.75rem;
        color: #3a3a48;
        letter-spacing: 0.08em;
    }

    /* ── Warning / error ── */
    .stWarning, .stException {
        background: #1a1510 !important;
        border-color: #5a4010 !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── ML helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def load_word_index():
    raw = imdb.get_word_index()
    wi = {w: i + 3 for w, i in raw.items()}
    wi["<PAD>"] = 0; wi["<START>"] = 1; wi["<UNK>"] = 2; wi["<UNUSED>"] = 3
    return wi

def preprocess_review(review, word_index, maxlen, vocab_size):
    review = re.sub(r"[^a-z0-9\s']", "", review.lower())
    seq = [1] + [min(word_index.get(w, 2), vocab_size - 1) for w in review.split()]
    return pad_sequences([seq], maxlen=maxlen)

def build_model(vocab_size):
    m = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LENGTH),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

@st.cache_resource
def load_sentiment_model():
    if os.path.exists(MODEL_PATH):
        try:
            return load_model(MODEL_PATH)
        except Exception:
            pass
    st.info("First launch: training the model (~2–3 min). Grab a coffee ☕")
    (X_train, y_train), _ = imdb.load_data(num_words=VOCAB_SIZE)
    X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
    model = build_model(VOCAB_SIZE)
    bar = st.progress(0, text="Training epoch 1/3…")
    for ep in range(3):
        model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=0)
        bar.progress(int((ep + 1) / 3 * 100), text=f"Training epoch {ep + 1}/3…")
    bar.empty()
    model.save(MODEL_PATH)
    return model


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="CineScope — Sentiment Analyser",
        page_icon="🎬",
        layout="centered"
    )
    inject_css()

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">AI · NLP · Deep Learning</div>
        <div class="hero-title">Cine<span>Scope</span></div>
        <div class="hero-subtitle">
            Paste any Movie detailed review  and our LSTM model instantly reads the emotion behind the words.
        </div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    try:
        model = load_sentiment_model()
    except Exception as exc:
        st.error(f"Model error: {exc}")
        return

    word_index = load_word_index()

    # Input card
    st.markdown('<div class="input-card"><div class="input-label">Your Review</div>', unsafe_allow_html=True)
    review_text = st.text_area(
        label="review",
        placeholder="e.g. The cinematography was breathtaking, but the pacing dragged in the second act…",
        height=160,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    analyse = st.button("Analyse Sentiment")

    if analyse:
        if not review_text.strip():
            st.warning("Please enter some review text first.")
        else:
            with st.spinner("Reading between the lines…"):
                seq = preprocess_review(review_text, word_index, MAX_LENGTH, VOCAB_SIZE)
                prob = float(model.predict(seq, verbose=0)[0][0])

            is_pos = prob >= 0.5
            sentiment_word = "POSITIVE" if is_pos else "NEGATIVE"
            sentiment_class = "result-sentiment-positive" if is_pos else "result-sentiment-negative"
            confidence = prob if is_pos else 1 - prob
            bar_pct = int(prob * 100)
            bar_class = "prob-bar-fill-pos" if is_pos else "prob-bar-fill-neg"

            word_count = len(review_text.split())
            tone = "Confident" if confidence > 0.80 else "Moderate" if confidence > 0.60 else "Borderline"

            # Stats row
            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-value">{word_count}</div>
                    <div class="stat-key">Words</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{confidence:.0%}</div>
                    <div class="stat-key">Confidence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{tone}</div>
                    <div class="stat-key">Signal</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Result card
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Sentiment</div>
                <div class="{sentiment_class}">{sentiment_word}</div>
                <div class="result-confidence">
                    Model confidence: <strong>{confidence:.1%}</strong>
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-labels">
                        <span>Negative</span>
                        <span>Positive</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="{bar_class}" style="width:{bar_pct}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        LSTM · IMDB 25k dataset · Keras / TensorFlow · Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
