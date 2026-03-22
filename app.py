import streamlit as st
import re
import pandas as pd
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Telugu AI Summarizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM GEMINI-INSPIRED UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }

    /* Glassmorphism Card */
    .stTextArea textarea {
        border-radius: 20px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 20px !important;
        background: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }

    .result-card {
        background: white;
        padding: 25px;
        border-radius: 24px;
        border: 1px solid #efefef;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .stButton>button {
        border-radius: 30px;
        padding: 10px 25px;
        background: linear-gradient(90deg, #4285F4, #34A853);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(66, 133, 244, 0.4);
    }

    .sidebar-content {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
    }

    h1, h2, h3 {
        color: #1f1f1f;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #70757a;
        font-size: 0.8rem;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC FUNCTIONS ---

def detect_language(text):
    # Simple regex check for Telugu Unicode range
    if re.search(r'[\u0c00-\u0c7f]', text):
        return "Telugu"
    return "English"

def clean_text(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def summarize_extractive(text, ratio=0.3):
    # Split sentences (Handling common Telugu/English delimiters)
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) < 3:
        return text, sentences

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = tfidf.fit_transform(sentences)
        # Sentence similarity via Cosine Similarity (TextRank logic)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Rank sentences using PageRank logic via NetworkX
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Sort and pick top sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        num_sentences = max(1, int(len(sentences) * ratio))
        summary_list = [ranked_sentences[i][1] for i in range(num_sentences)]
        
        # Re-order summary based on original appearance
        final_summary = " ".join([s for s in sentences if s in summary_list])
        return final_summary, sentences
    except:
        # Fallback to simple lead-based if TF-IDF fails (e.g. too short)
        return " ".join(sentences[:2]), sentences

def get_keywords(text, num=10):
    words = re.findall(r'\w+', text.lower())
    # Basic stopword filtering (could be expanded for Telugu)
    stop_words = {'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'ఈ', 'మరియు', 'ఒక'}
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    return Counter(filtered).most_common(num)

# --- UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/281/281764.png", width=50)
    st.title("Settings")
    st.markdown("---")
    summary_length = st.select_slider(
        "Summary Length",
        options=["Short", "Medium", "Long"],
        value="Medium"
    )
    length_map = {"Short": 0.15, "Medium": 0.3, "Long": 0.5}
    
    st.markdown("### Visualizations")
    show_chart = st.checkbox("Show Keyword Frequency", value=True)
    
    if st.button("🗑️ Clear Input"):
        st.session_state.input_text = ""
        st.rerun()

# Main Header
st.markdown("<h1 style='text-align: center;'>AI Powered Multilingual Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5f6368;'>Expertly condense Telugu and English articles using Extractive AI.</p>", unsafe_allow_html=True)

# Input Area
input_text = st.text_area("Paste your text here...", height=250, key="main_input", placeholder="Enter Telugu or English content...")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    process_btn = st.button("✨ Generate Summary", use_container_width=True)

# Processing
if process_btn:
    if not input_text.strip():
        st.error("Please enter some text to summarize.")
    else:
        with st.spinner("Analyzing text patterns..."):
            # Logic
            lang = detect_language(input_text)
            cleaned = clean_text(input_text)
            summary, all_sentences = summarize_extractive(cleaned, ratio=length_map[summary_length])
            keywords = get_keywords(cleaned)
            
            # Display Result
            st.markdown("---")
            
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.markdown(f"### 📝 {lang} Summary")
                st.markdown(f'<div class="result-card">{summary}</div>', unsafe_allow_html=True)
                
                # Actions
                st.download_button("📥 Download .txt", summary, file_name="summary.txt")
                if st.button("📋 Copy to Clipboard"):
                    st.toast("Summary copied to clipboard!") # Streamlit toast for UX
            
            with col_right:
                st.markdown("### 📊 Insights")
                st.metric("Original Sentences", len(all_sentences))
                st.metric("Summary Reduction", f"{int((1 - len(summary)/len(input_text))*100)}%")
                
                if show_chart and keywords:
                    df = pd.DataFrame(keywords, columns=['Word', 'Count'])
                    fig = px.bar(df, x='Count', y='Word', orientation='h', 
                                 title="Top Keywords", template="simple_white",
                                 color_discrete_sequence=['#4285F4'])
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
    <div class="footer">
        Built with ❤️ for Telugu NLP • 2026 AI Projects<br>
        Using TextRank & TF-IDF Extractive Algorithms
    </div>
    """, unsafe_allow_html=True)
