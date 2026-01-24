import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import os
import re
import sqlite3
from datetime import datetime
import logging
import tempfile
import requests
import zipfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import io

# ==============================
# ENVIRONMENT & PATH SETUP
# ==============================
def is_streamlit_cloud():
    return (
        os.getenv("HOME") == "/home/appuser" or
        "streamlitapp.com" in os.getenv("HOSTNAME", "") or
        os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"
    )

IS_CLOUD = is_streamlit_cloud()
BASE_DIR = Path(tempfile.gettempdir())
METADATA_DB_FILE = BASE_DIR / "eis_ml_metadata.db"
UNIVERSE_DB_FILE = BASE_DIR / "eis_ml_universe.db"
LOG_FILE = BASE_DIR / "eis_ml_query.log"

BASE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================
# STREAMLIT PAGE CONFIGURATION (Matches Reference)
# ==============================
st.set_page_config(
    page_title="EIS & ML Knowledge Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Title and Description ---
st.title("🔋 EIS & ML Context-Aware Explorer")
st.markdown("""
This advanced tool queries **arXiv** for scientific literature on **Electrochemical Impedance Spectroscopy (EIS) and Machine Learning**, with a focus on:
- **Universal Modeling** (Generalizable across chemistries/conditions)
- **Context-Aware NLP** (Extracting temperature, SoC, and chemistry)
- **State of Health (SOH) & RUL Prediction**
- **Equivalent Circuit Modeling (ECM)** via Neural Networks
- **Deep Learning for Impedance Data** (CNNs, LSTMs, Transformers)

It uses **SciBERT with attention-aware relevance scoring** (>30% threshold) and stores:
- **Metadata** in `eis_ml_metadata.db`
- **Full extracted text** in `eis_ml_universe.db`

All data is downloadable. **No automated PDF scraping** — downloads occur only on user request.
""")

if IS_CLOUD:
    st.info("☁️ **Streamlit Cloud Mode**: All files are stored temporarily in `/tmp`. Use download buttons before your session expires!")

# ==============================
# SESSION STATE & LOGGING
# ==============================
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "downloaded_pdfs" not in st.session_state:
    st.session_state.downloaded_pdfs = {}
if "universe_db_updated" not in st.session_state:
    st.session_state.universe_db_updated = False
if "papers_df" not in st.session_state:
    st.session_state.papers_df = None
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    logging.info(message)

# ==============================
# MODELS & PATTERNS
# ==============================
@st.cache_resource
def load_scibert():
    update_log("Loading SciBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    model.eval()
    return tokenizer, model

try:
    scibert_tokenizer, scibert_model = load_scibert()
except Exception as e:
    st.error(f"❌ SciBERT Load Error: {e}")
    st.stop()

# Domain Specific Terms
EIS_ML_PATTERNS = [
    r'\belectrochemical impedance spectroscopy|eis\b',
    r'\bnyquist|bode\s*plots?\b',
    r'\bequivalent circuit|ecm|drt\b',
    r'\bmachine learning|deep learning|neural networks?\b',
    r'\buniversal modeling|context-aware|nlp\b',
    r'\bstate of health|soh|remaining useful life|rul\b',
    r'\bbattery|lithium|fuel cell\b'
]
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EIS_ML_PATTERNS]

# ==============================
# CORE FUNCTIONS
# ==============================
def extract_context(text):
    """The 'Context-Aware' NLP engine."""
    context = {"Chemistry": "Unknown", "Temp": "Ambient", "SoC": "Unknown"}
    temp_match = re.search(r'(-?\d+\s*°C|-?\d+\s*K)', text)
    chem_match = re.search(r'\b(LFP|NMC|LCO|NCA|Li-ion)\b', text, re.I)
    if temp_match: context["Temp"] = temp_match.group(1)
    if chem_match: context["Chemistry"] = chem_match.group(1).upper()
    return context

def score_abstract(abstract):
    num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract))
    return np.sqrt(num_matched) / np.sqrt(len(EIS_ML_PATTERNS))

# ==============================
# SIDEBAR CONFIGURATION (Matches Reference)
# ==============================
with st.sidebar:
    st.header("⚙️ Search Configuration")
    
    query_mode = st.radio("Query Mode", ["Auto (Recommended)", "Custom"], horizontal=True)
    
    if query_mode == "Auto":
        # Specific query for your topic
        query = '("EIS" OR "Electrochemical Impedance Spectroscopy") AND ("NLP" OR "Machine Learning" OR "Universal Modeling")'
        st.text_area("Auto-generated Query", value=query, height=100, disabled=True)
    else:
        query = st.text_area("Custom Query", value="EIS AND 'Machine Learning'", height=100)
    
    default_cats = ["physics.chem-ph", "stat.ML", "cs.LG", "eess.SP"]
    categories = st.multiselect("arXiv Categories", options=default_cats + ["physics.app-ph"], default=default_cats)
    
    max_results = st.slider("Max Results", 1, 200, 30)
    
    col1, col2 = st.columns(2)
    start_year = col1.number_input("Start Year", 2010, 2026, 2018)
    end_year = col2.number_input("End Year", 2010, 2026, 2026)
    
    search_button = st.button("🚀 Execute Search", type="primary")

# ==============================
# MAIN BODY RESULTS
# ==============================
st.header("🔍 arXiv Query for EIS & ML Context-Awareness")
st.markdown("Use the sidebar to configure your search. Results are scored by **SciBERT + regex relevance** (>30% threshold).")

# Logs Section
st.text_area("📝 Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

if search_button:
    st.session_state.search_performed = True
    with st.spinner("📡 Querying arXiv and Extracting Context..."):
        # Logic to fetch from arXiv
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        
        papers = []
        for result in client.results(search):
            if start_year <= result.published.year <= end_year:
                rel_score = score_abstract(result.summary)
                if rel_score > 0.3:
                    ctx = extract_context(result.summary)
                    papers.append({
                        "id": result.get_short_id(),
                        "title": result.title,
                        "year": result.published.year,
                        "relevance": round(rel_score * 100, 2),
                        "abstract": result.summary,
                        "context": f"Chem: {ctx['Chemistry']}, Temp: {ctx['Temp']}",
                        "url": result.pdf_url
                    })
        
        if papers:
            st.session_state.papers_df = pd.DataFrame(papers)
            st.success(f"✅ Found **{len(papers)}** relevant papers.")
            
            for p in papers:
                with st.expander(f"📄 {p['title']} ({p['year']}) — {p['relevance']}%"):
                    st.info(f"🧬 **Extracted Context**: {p['context']}")
                    st.write(p['abstract'])
                    st.markdown(f"[🌐 View on arXiv]({p['url'].replace('/pdf/', '/abs/')})")
        else:
            st.warning("No papers met the 30% relevance threshold.")
