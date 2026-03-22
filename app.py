# app.py
# AI Powered Multilingual Text Summarizer

import streamlit as st
import nltk
import string
from collections import Counter
import matplotlib.pyplot as plt
from langdetect import detect

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# --------------------------
# CUSTOM CSS (Gemini-like UI)
# --------------------------
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
.main {
    padding: 2rem;
}
.stTextArea textarea {
    border-radius: 12px;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
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
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word not in string.punctuation]

    return sentences, words


def summarize_text(text, ratio=0.3):
    sentences, words = preprocess_text(text, 'en')

    word_freq = Counter(words)

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    # Sort sentences
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    select_length = int(len(sentences) * ratio)
    summary = ranked_sentences[:select_length]

    return " ".join(summary), word_freq, sentences


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
    body { background-color: #0E1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# MAIN UI
# --------------------------

st.title("✨ AI Powered Multilingual Text Summarizer")
st.markdown("""
Summarize Telugu and English text using extractive NLP techniques.
""")

user_input = st.text_area("Enter your text here:", height=250)

col1, col2, col3 = st.columns(3)

with col1:
    summarize_btn = st.button("Summarize")

with col2:
    clear_btn = st.button("Clear")

with col3:
    download_btn = st.button("Download Summary")

# --------------------------
# LOGIC
# --------------------------

if clear_btn:
    user_input = ""

if summarize_btn:
    if not user_input.strip():
        st.error("Please enter some text!")
    else:
        with st.spinner("Processing..."):
            lang = detect_language(user_input)
            summary, word_freq, sentences = summarize_text(user_input, summary_length)
            keywords = extract_keywords(word_freq)

        st.success(f"Detected Language: {lang}")

        st.subheader("📄 Original Text")
        st.write(user_input)

        st.subheader("📝 Summary")
        st.write(summary)

        # Keywords
        st.subheader("🔑 Keywords")
        for word, freq in keywords:
            st.write(f"{word} ({freq})")

        # Word Frequency Chart
        st.subheader("📊 Word Frequency")
        words = [w for w, f in keywords]
        freqs = [f for w, f in keywords]

        fig, ax = plt.subplots()
        ax.bar(words, freqs)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Download
        if download_btn:
            st.download_button("Download", summary, file_name="summary.txt")

# --------------------------
# FOOTER
# --------------------------
st.markdown("""
<div class='footer'>
Built with ❤️ using Streamlit | Telugu NLP Project
</div>
""", unsafe_allow_html=True)

# --------------------------
# REQUIREMENTS (for reference)
# --------------------------
# streamlit
# nltk
# matplotlib
# langdetect
