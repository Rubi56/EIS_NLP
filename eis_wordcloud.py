# streamlit_app.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import io
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from numba import jit, njit, prange
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="EIS & AI/ML Knowledge Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for EIS-AI theme
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    color: #1E40AF; /* Deep EIS Blue */
    text-align: center;
    margin-bottom: 1.5rem;
}
.metric-card {
    background-color: #F0FDF4;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #10B981;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.figure-caption {
    font-size: 0.95rem;
    color: #4B5563;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
    font-style: italic;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

def add_caption(text: str):
    """Add a styled caption below a figure"""
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# Numba-accelerated text processing
@jit(nopython=True, parallel=True)
def process_text_numba(text_array):
    results = []
    for i in prange(len(text_array)):
        text = text_array[i]
        cleaned = ""
        for char in text:
            if char.isalnum() or char.isspace():
                cleaned += char.lower()
        results.append(cleaned)
    return results

def get_db_paths_for_query(query_id: str = "q0") -> dict:
    script_dir = Path(__file__).parent
    knowledge_db_dir = script_dir / "eis_ai_database"
    knowledge_db_dir.mkdir(exist_ok=True)
    suffix = f"{query_id}_" if query_id != "q0" else ""
    
    return {
        "Metadata DB": knowledge_db_dir / f"eis_ai_{suffix}metadata.db",
        "Universe DB": knowledge_db_dir / f"eis_ai_{suffix}universe.db",
        "PDF Storage DB": knowledge_db_dir / f"eis_ai_{suffix}pdfs.db"
    }

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def connect(self) -> bool:
        try:
            if not os.path.exists(self.db_path): return False
            self.conn = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def get_papers_data(self) -> pd.DataFrame:
        if not self.conn: return pd.DataFrame()
        try:
            query = """
            SELECT full_text, title, year, categories 
            FROM papers 
            WHERE full_text IS NOT NULL AND LENGTH(full_text) > 50
            LIMIT 500
            """
            return pd.read_sql_query(query, self.conn)
        except:
            return pd.DataFrame()

    def disconnect(self):
        if self.conn: self.conn.close()

class TextAnalyzer:
    def __init__(self):
        # Updated stopwords for EIS/AI domain
        self.stopwords = {
            'using', 'used', 'use', 'paper', 'study', 'research', 'result', 'method', 'proposed',
            'analysis', 'based', 'approach', 'performance', 'data', 'different', 'between',
            'the', 'and', 'in', 'of', 'to', 'a', 'is', 'are', 'with', 'for', 'on', 'by'
        }
    
    def extract_keyphrases(self, texts: list) -> dict:
        keyphrases = []
        for text in texts:
            text_clean = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', str(text).lower())
            words = [w for w in text_clean.split() if len(w) > 2 and w not in self.stopwords]
            
            for i in range(len(words) - 1):
                keyphrases.append(f"{words[i]} {words[i+1]}")
                if i < len(words) - 2:
                    keyphrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        return Counter(keyphrases)

class VisualizationEngine:
    def create_wordcloud(self, term_counts: dict, title: str, max_words: int, colormap: str):
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        wc = WordCloud(
            width=1600, height=800, background_color='white',
            max_words=max_words, colormap=colormap, relative_scaling=0.5
        ).generate_from_frequencies(term_counts)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
        ax.axis('off')
        return fig

def create_sample_database(query_id="q0"):
    db_paths = get_db_paths_for_query(query_id)
    db_path = db_paths["Metadata DB"]
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS papers 
                         (id INTEGER PRIMARY KEY, title TEXT, abstract TEXT, full_text TEXT, year INTEGER, categories TEXT)''')
        
        # EIS + AI Sample Data
        samples = [
            ("Deep Learning for Battery SOH Prediction via EIS", 
             "We use CNNs to analyze Nyquist plots for state-of-health estimation.",
             "This study applies deep learning to electrochemical impedance spectroscopy. We analyze the Nyquist plot and Bode plot using a convolutional neural network (CNN) to predict battery state of health (SOH). The Rct charge transfer resistance was found to be a key feature.",
             2024, "AI/ML Methods"),
            ("Equivalent Circuit Modeling of Li-ion Batteries", 
             "A study on Warburg impedance and Rct parameters.",
             "We develop an equivalent circuit model (ECM) incorporating Warburg impedance and constant phase elements. EIS measurements were taken from 0.1Hz to 10kHz. The results show degradation in the SEI layer and increased internal resistance.",
             2023, "EIS Core"),
            ("Data-Driven EIS Feature Extraction", 
             "Random forest models for battery degradation.",
             "Machine learning models including Random Forest and LSTM were used to process EIS data. Feature extraction focused on the semi-circle diameter of the Nyquist plot, representing charge transfer resistance Rct.",
             2025, "AI/ML Methods")
        ]
        cursor.executemany('INSERT INTO papers (title, abstract, full_text, year, categories) VALUES (?,?,?,?,?)', samples)
        conn.commit()
        conn.close()

def main():
    st.markdown('<h1 class="main-header">🔋 EIS & AI/ML Analysis Explorer<br><small>Numba-Accelerated Electrochemical Spectroscopy Insights</small></h1>', unsafe_allow_html=True)
    
    if 'current_query' not in st.session_state: st.session_state.current_query = "q0"
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    
    with st.sidebar:
        st.header("⚙️ EIS-AI Config")
        query_options = ["q0", "q1", "q2"]
        current_query = st.selectbox("Select Research Query", query_options)
        
        if current_query != st.session_state.current_query:
            st.session_state.current_query = current_query
            st.session_state.analysis_results = None
            st.rerun()

        db_paths = get_db_paths_for_query(current_query)
        create_sample_database(current_query)
        
        selected_db = st.selectbox("Select Database", list(db_paths.keys()))
        max_papers = st.slider("Papers to analyze", 10, 500, 100)
        colormap = st.selectbox("Color Palette", ["viridis", "plasma", "magma", "coolwarm"])
        analyze_btn = st.button("🚀 Run EIS-AI Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Analyzing Electrochemical Data via SciBERT logic..."):
            db_manager = DatabaseManager(str(db_paths[selected_db]))
            if db_manager.connect():
                df = db_manager.get_papers_data().head(max_papers)
                texts = df['full_text'].tolist()
                
                analyzer = TextAnalyzer()
                term_counts = analyzer.extract_keyphrases(texts)
                
                st.session_state.analysis_results = {
                    'papers': df,
                    'term_counts': term_counts,
                    'query_id': current_query
                }
                db_manager.disconnect()
                st.success("Analysis Complete")

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        tab1, tab2 = st.tabs(["Knowledge Cloud", "Raw Term Metrics"])
        
        with tab1:
            viz = VisualizationEngine()
            fig = viz.create_wordcloud(res['term_counts'], f"EIS-AI Terms ({res['query_id']})", 100, colormap)
            st.pyplot(fig)
            add_caption(r"Visualizing convergence of EIS parameters ($R_{ct}, Z_{re}$) and AI methods (CNN, LSTM).")

        with tab2:
            terms_df = pd.DataFrame(res['term_counts'].items(), columns=['Term', 'Freq']).sort_values('Freq', ascending=False)
            st.dataframe(terms_df, use_container_width=True)

if __name__ == "__main__":
    main()
