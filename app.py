# app.py
# AI Powered Multilingual Text Summarizer

import streamlit as st
import string
from collections import Counter
import matplotlib.pyplot as plt
from langdetect import detect
import nltk
import os

# --------------------------
# NLTK SETUP (STREAMLIT SAFE)
# --------------------------
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# --------------------------
# CUSTOM CSS (Modern UI)
# --------------------------
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}
textarea {
    border-radius: 12px !important;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.footer {
    text-align: center;
    padding: 20px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# FUNCTIONS
# --------------------------

def detect_language(text):
    try:
        lang = detect(text)
        return "Telugu" if lang == 'te' else "English"
    except:
        return "Unknown"


def preprocess_text(text, lang):
    # Sentence splitting (safe)
    if lang == "Telugu":
        sentences = text.split(".")
    else:
        sentences = text.replace("!", ".").replace("?", ".").split(".")

    sentences = [s.strip() for s in sentences if s.strip()]

    # Word tokenization
    words = text.lower().split()

    # Stopwords
    telugu_stopwords = ["ఈ", "ఆ", "లో", "కు", "పై", "కోసం", "తో", "ని"]

    if lang == "Telugu":
        stop_words = set(telugu_stopwords)
    else:
        stop_words = set(stopwords.words('english'))

    words = [
        word for word in words
        if word not in stop_words and word not in string.punctuation
    ]

    return sentences, words


def summarize_text(text, ratio=0.3, lang="English"):
    sentences, words = preprocess_text(text, lang)

    word_freq = Counter(words)

    sentence_scores = {}

    for sentence in sentences:
        words_in_sentence = sentence.lower().split()

        for word in words_in_sentence:
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    # Rank sentences
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    select_length = max(1, int(len(sentences) * ratio))
    summary_sentences = ranked_sentences[:select_length]

    return " ".join(summary_sentences), word_freq, sentences, summary_sentences


def extract_keywords(word_freq, n=10):
    return word_freq.most_common(n)


# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("⚙️ Controls")

summary_length = st.sidebar.slider("Summary Length", 0.1, 1.0, 0.3)

mode = st.sidebar.radio("Theme", ["Light", "Dark"])

if mode == "Dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    textarea {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# MAIN UI
# --------------------------

st.title("✨ AI Powered Multilingual Text Summarizer")
st.markdown("Summarize Telugu and English text using extractive NLP techniques.")

user_input = st.text_area("Enter your text here:", height=250)

col1, col2 = st.columns(2)

with col1:
    summarize_btn = st.button("🚀 Summarize")

with col2:
    clear_btn = st.button("🧹 Clear")

# --------------------------
# LOGIC
# --------------------------

if clear_btn:
    st.session_state["text"] = ""
    st.experimental_rerun()

if summarize_btn:
    if not user_input.strip():
        st.error("⚠️ Please enter some text!")
    else:
        with st.spinner("⏳ Processing..."):
            lang = detect_language(user_input)

            summary, word_freq, sentences, summary_sentences = summarize_text(
                user_input, summary_length, lang
            )

            keywords = extract_keywords(word_freq)

        st.success(f"🌐 Detected Language: {lang}")

        # Original Text
        st.subheader("📄 Original Text")
        for sentence in sentences:
            if sentence in summary_sentences:
                st.markdown(f"**🟢 {sentence}**")
            else:
                st.write(sentence)

        # Summary
        st.subheader("📝 Summary")
        st.write(summary)

        # Keywords
        st.subheader("🔑 Keywords")
        st.write(", ".join([word for word, _ in keywords]))

        # Word Frequency Chart
        st.subheader("📊 Word Frequency")
        words = [w for w, _ in keywords]
        freqs = [f for _, f in keywords]

        fig, ax = plt.subplots()
        ax.bar(words, freqs)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Download Button (FIXED)
        st.download_button(
            "📥 Download Summary",
            summary,
            file_name="summary.txt"
        )

# --------------------------
# FOOTER
# --------------------------
st.markdown("""
<div class='footer'>
✨ Built with Streamlit | Telugu NLP Extractive Summarizer Project
</div>
""", unsafe_allow_html=True)
