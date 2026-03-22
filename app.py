import streamlit as st
import streamlit.components.v1 as components
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# --- CONFIG ---
st.set_page_config(page_title="AI Summarizer Pro", layout="wide")

# --- BACKEND LOGIC ---
def summarize_logic(text, ratio=0.3):
    if len(text) < 50: return "Text too short to summarize.", []
    
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) < 3: return text, sentences
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    num_sents = max(1, int(len(sentences) * ratio))
    summary_list = [ranked[i][1] for i in range(num_sents)]
    
    final_summary = " ".join([s for s in sentences if s in summary_list])
    keywords = re.findall(r'\w+', text.lower())[:10] # Simplified for demo
    return final_summary, keywords

# --- CUSTOM HTML/CSS/JS FRONTEND ---
# This is a Gemini-inspired "Prompt Box" UI
html_code = """
<!DOCTYPE html>
<html>
<head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #fdfdfd; margin: 0; display: flex; justify-content: center; }
        .container { width: 90%; max-width: 800px; margin-top: 50px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-weight: 600; color: #1f1f1f; font-size: 2.5rem; letter-spacing: -1px; }
        
        .chat-input-container {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 28px;
            padding: 15px 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            transition: 0.3s;
            display: flex;
            flex-direction: column;
        }
        .chat-input-container:focus-within {
            box-shadow: 0 4px 25px rgba(0,0,0,0.1);
            border-color: #4285f4;
        }
        textarea {
            width: 100%;
            border: none;
            outline: none;
            font-size: 16px;
            resize: none;
            min-height: 100px;
            color: #3c4043;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        .btn-summarize {
            background: #1a73e8;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s;
        }
        .btn-summarize:hover { background: #1557b0; }
        
        .result-card {
            margin-top: 30px;
            background: #f8f9fa;
            border-radius: 24px;
            padding: 30px;
            border: 1px solid #eee;
            line-height: 1.6;
            color: #1f1f1f;
            display: none; /* Hidden until processing */
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Summarize anything.</h1>
        </div>

        <div class="chat-input-container">
            <textarea id="textInput" placeholder="Paste your Telugu or English text here..."></textarea>
            <div class="controls">
                <span style="color: #70757a; font-size: 12px;">Extractive Method • Multilingual</span>
                <button class="btn-summarize" onclick="sendData()">Summarize</button>
            </div>
        </div>

        <div id="resultBox" class="result-card">
            <div style="font-weight: 600; margin-bottom: 10px; color: #4285f4;">AI Summary</div>
            <div id="summaryText"></div>
        </div>
    </div>

    <script>
        function sendData() {
            const text = document.getElementById('textInput').value;
            if(!text) return;
            
            // Communicate with Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: text
            }, '*');
        }
        
        // Listen for summary from Python
        window.addEventListener('message', function(event) {
            if (event.data.type === 'summaryUpdate') {
                document.getElementById('resultBox').style.display = 'block';
                document.getElementById('summaryText').innerText = event.data.summary;
            }
        });
    </script>
</body>
</html>
"""

# --- RENDER & BRIDGE ---
# We use a trick: st.text_area is hidden, and HTML JS triggers the process
input_val = components.html(html_code, height=600)

# In a real app, you'd use a more complex bridge, but for a single file:
# Let's use standard Streamlit for the trigger but style it heavily.

# Since Streamlit's component bridge is one-way (JS -> Py), 
# I will use a hybrid approach where Python handles the "Result Card" rendering 
# to ensure the summary is actually visible and downloadable.

st.markdown("""
    <style>
    .stTextArea, .stButton { display: none; } /* Hide the ugly default parts */
    </style>
""", unsafe_allow_html=True)

# Custom Input Area
raw_text = st.text_area("Hidden Input", key="hidden_input")

if raw_text:
    with st.spinner(" "):
        summary, keywords = summarize_logic(raw_text)
        
        # Displaying the result in a modern container
        st.markdown(f"""
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <div style="background: white; border-radius: 24px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); border: 1px solid #f0f0f0;">
                    <h3 style="color: #4285f4; margin-top: 0;">✨ Summary</h3>
                    <p style="font-size: 1.1rem; line-height: 1.7; color: #3c4043;">{summary}</p>
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                    <div style="display: flex; gap: 10px;">
                         <span style="background: #e8f0fe; color: #1967d2; padding: 5px 15px; border-radius: 15px; font-size: 12px;">Extractive AI</span>
                         <span style="background: #e6f4ea; color: #137333; padding: 5px 15px; border-radius: 15px; font-size: 12px;">Telugu Supported</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.download_button("Download Summary", summary, file_name="summary.txt")
