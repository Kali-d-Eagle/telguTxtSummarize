# app.py
# Advanced AI Multilingual Text Summarizer

import streamlit as st
import string
from collections import Counter
from langdetect import detect
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="AI Summarizer", layout="wide")

# --------------------------
# MODERN UI CSS (Glassmorphism)
# --------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}
textarea {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# FUNCTIONS
# --------------------------

def detect_language(text):
    try:
        lang = detect(text)
        return "Telugu" if lang == "te" else "English"
    except:
        return "Unknown"


def split_sentences(text):
    text = text.replace("!", ".").replace("?", ".")
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences


def summarize_text_tfidf(text, ratio=0.3):
    sentences = split_sentences(text)

    if len(sentences) < 2:
        return text, [], []

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Sentence scores
    scores = similarity_matrix.sum(axis=1)

    ranked_sentences = [
        sentences[i] for i in np.argsort(scores)[::-1]
    ]

    select_length = max(1, int(len(sentences) * ratio))
    summary = ranked_sentences[:select_length]

    return " ".join(summary), sentences, summary


def extract_keywords(text, n=10):
    words = text.lower().split()
    words = [w for w in words if w not in string.punctuation]
    freq = Counter(words)
    return freq.most_common(n)


# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("⚙️ Controls")

ratio = st.sidebar.slider("Summary Length", 0.1, 1.0, 0.3)

theme = st.sidebar.radio("Theme", ["Dark", "Light"])

# --------------------------
# HEADER
# --------------------------
st.title("✨ AI Powered Multilingual Text Summarizer")
st.markdown("Advanced Extractive NLP + AI-inspired UI")

# --------------------------
# CHAT UI
# --------------------------
user_input = st.text_area("💬 Enter your text...", height=200)

if st.button("🚀 Generate Summary"):

    if not user_input.strip():
        st.error("Enter text first!")
    else:
        with st.spinner("Analyzing text... 🤖"):
            lang = detect_language(user_input)
            summary, sentences, important = summarize_text_tfidf(user_input, ratio)
            keywords = extract_keywords(user_input)

        # --------------------------
        # OUTPUT
        # --------------------------
        st.success(f"Detected Language: {lang}")

        # Chat Style UI
        st.markdown("### 💬 Conversation")

        st.markdown(f"""
        <div class="card">
        <b>🧑 You:</b><br>{user_input}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
        <b>🤖 AI Summary:</b><br>{summary}
        </div>
        """, unsafe_allow_html=True)

        # Highlight Sentences
        st.subheader("📄 Important Sentences")
        for s in sentences:
            if s in important:
                st.markdown(f"🟢 **{s}**")
            else:
                st.write(s)

        # Keywords
        st.subheader("🔑 Keywords")
        st.write(", ".join([w for w, _ in keywords]))

        # --------------------------
        # Plotly Chart
        # --------------------------
        st.subheader("📊 Word Frequency (Interactive)")
        words = [w for w, _ in keywords]
        freqs = [f for _, f in keywords]

        fig = px.bar(x=words, y=freqs)
        st.plotly_chart(fig)

        # --------------------------
        # WordCloud
        # --------------------------
        st.subheader("☁️ Word Cloud")
        wc = WordCloud(width=800, height=400).generate(user_input)

        fig2, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig2)

        # Download
        st.download_button("📥 Download Summary", summary, "summary.txt")

# --------------------------
# FOOTER
# --------------------------
st.markdown("""
---
✨ Built with Streamlit | Advanced NLP Summarizer Project
""")
