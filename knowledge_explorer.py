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
    """Detect if running on Streamlit Cloud."""
    return (
        os.getenv("HOME") == "/home/appuser" or
        "streamlitapp.com" in os.getenv("HOSTNAME", "") or
        os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"
    )

IS_CLOUD = is_streamlit_cloud()

# Use a guaranteed-writable temporary directory
BASE_DIR = Path(tempfile.gettempdir())
METADATA_DB_FILE = BASE_DIR / "eis_ai_metadata.db"
UNIVERSE_DB_FILE = BASE_DIR / "eis_ai_universe.db"
LOG_FILE = BASE_DIR / "eis_ai_query.log"

BASE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================
# STREAMLIT PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="EIS + AI Knowledge Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔋 EIS & Artificial Intelligence Explorer")
st.markdown("""
This tool queries **arXiv** for the intersection of **Electrochemical Impedance Spectroscopy (EIS)** and **Artificial Intelligence**.
- **Mandatory Focus**: EIS, Impedance Spectra, Nyquist/Bode Analysis.
- **AI Integration**: Machine Learning (ML), Deep Learning, Physics-Informed Neural Networks (PINNs), Data-driven modeling, and NLP for spectral mining.
- **Scoring**: Uses **SciBERT** with attention-aware scoring specifically tuned for electrochemical data science.
""")

if IS_CLOUD:
    st.info("☁️ **Cloud Mode**: Files stored in `/tmp`. Download your databases before the session ends!")

# ==============================
# SESSION STATE INITIALIZATION
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
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# ==============================
# SCIBERT MODEL LOADING
# ==============================
@st.cache_resource
def load_scibert():
    update_log("Loading SciBERT (tuned for EIS/AI detection)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        return tokenizer, model
    except Exception as e:
        update_log(f"Failed to load SciBERT: {e}")
        raise e

try:
    scibert_tokenizer, scibert_model = load_scibert()
except Exception as e:
    st.error(f"❌ SciBERT Load Error: {e}")
    st.stop()

# ==============================
# EIS & AI PATTERN DEFINITIONS
# ==============================
@st.cache_data
def normalize_text(text):
    # Standard normalization plus Ohm/Sigma for EIS context
    greek_to_latin = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'Ω': 'ohm', 'σ': 'sigma', 'π': 'pi'
    }
    for greek, latin in greek_to_latin.items():
        text = text.replace(greek, latin)
    return text.lower()

# Expanded core terms (EIS) - more variations
EIS_CORE = [
    "EIS", "Electrochemical Impedance Spectroscopy", "Impedance Spectra", 
    "Nyquist", "Bode plot", "electrochemical impedance", "impedance spectroscopy",
    "electrochemical impedance analysis", "impedance measurement", "impedance data",
    "impedance response", "electrochemical impedance", "impedance analysis",
    "electrochemical analysis", "impedance spectrum", "impedance spectroscopy",
    "electrochemical system", "impedance modeling", "impedance characterization"
]

# Expanded AI/ML terms
AI_TERMS = [
    "Machine Learning", "Deep Learning", "Neural Network", "Artificial Intelligence", 
    "Data-driven", "Physics-informed", "NLP", "Natural Language Processing", 
    "Random Forest", "SVM", "CNN", "LSTM", "Transformers", "Gaussian Process", 
    "Bayesian optimization", "Predictive modeling", "Reinforcement learning",
    "Feature extraction", "Autoencoder", "AI", "ML", "DL", "PINN", "PINNs",
    "artificial neural network", "deep neural network", "convolutional neural network",
    "recurrent neural network", "supervised learning", "unsupervised learning",
    "regression", "classification", "clustering", "dimensionality reduction",
    "feature selection", "model training", "prediction", "forecasting"
]

# Expanded keywords for regex scoring
KEY_PATTERNS = [
    r'\belectrochemical impedance spectroscopy\b',
    r'\beis\b',
    r'\bimpedance (?:spectra|spectroscopy|measurements|data|analysis|response|modeling|characterization)\b',
    r'\bnyquist\b',
    r'\bbode\b',
    r'\bequivalent circuit model\b',
    r'\bmachine learning\b',
    r'\bdeep learning\b',
    r'\bneural networks?\b',
    r'\bdata[- ]driven\b',
    r'\bphysics[- ]informed\b',
    r'\bnatural language processing\b',
    r'\bnlp\b',
    r'\blarge language models?\b',
    r'\bconvolutional\b',
    r'\bpredictive (?:model|analytics)\b',
    r'\bsurrogate model\b',
    r'\bbayesian\b',
    r'\bstate of charge\b',
    r'\bstate of health\b',
    r'\bbattery (?:modeling|diagnosis|estimation)\b',
    r'\bfuel cell\b',
    r'\bsupercapacitor\b',
    r'\belectrolyte\b',
    r'\belectrode\b',
    r'\bcorrosion\b',
    r'\bbiosensor\b',
    r'\bimpedance tomography\b'
]

@st.cache_data
def compile_patterns():
    return [re.compile(pattern, re.IGNORECASE) for pattern in KEY_PATTERNS]

COMPILED_PATTERNS = compile_patterns()

# ==============================
# AI-BASED RELEVANCE SCORING
# ==============================
@st.cache_data
def score_abstract_with_scibert(abstract):
    """Score abstract relevance specifically for EIS + AI intersection."""
    try:
        inputs = scibert_tokenizer(
            abstract, return_tensors="pt", truncation=True, max_length=512, padding=True, return_attention_mask=True
        )
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        
        abstract_normalized = normalize_text(abstract)
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        
        # Mandatory Check: Must mention EIS or Spectroscopy
        is_eis = any(term.lower() in abstract_normalized for term in ["eis", "impedance", "spectroscopy", "nyquist", "bode"])
        # Mandatory Check: Must mention AI/Data-driven concepts
        is_ai = any(term.lower() in abstract_normalized for term in ["learning", "intelligence", "neural", "data", "driven", "model", "prediction", "ai", "ml", "dl"])
        
        base_score = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        
        if not is_eis: base_score *= 0.2  # Reduced penalty
        if not is_ai: base_score *= 0.7   # Reduced penalty
            
        # Attention boost for intersection keywords
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [
            i for i, token in enumerate(tokens)
            if any(kw in token.lower() for kw in ['impedance', 'eis', 'nyquist', 'ml', 'neural', 'learning', 'model', 'prediction'])
        ]
        
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.03:  # Lowered threshold
                base_score = min(base_score + 0.2, 1.0)  # Increased boost

        return base_score
    except:
        return 0.0

# ==============================
# DATABASE LOGIC
# ==============================
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if db_path == METADATA_DB_FILE:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, 
                categories TEXT, abstract TEXT, pdf_url TEXT, matched_terms TEXT, relevance_prob REAL
            )
        """)
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, content TEXT
            )
        """)
    conn.commit()
    conn.close()

# ==============================
# ARXIV QUERY (EIS + AI FOCUS)
# ==============================
@st.cache_data
def query_arxiv_eis_ai(categories, max_results, start_year, end_year):
    try:
        client = arxiv.Client()
        
        # Multiple search strategies to maximize results
        
        # Strategy 1: Broad query with OR conditions
        broad_query = '( "Electrochemical Impedance Spectroscopy" OR "EIS" OR "Impedance Spectra" OR "impedance spectroscopy" OR "electrochemical impedance" OR "impedance analysis" ) AND ' \
                     '( "Machine Learning" OR "Deep Learning" OR "Data-driven" OR "Neural Network" OR "Artificial Intelligence" OR "AI" OR "ML" OR "DL" OR "modeling" OR "prediction" )'
        
        # Strategy 2: Battery-focused query
        battery_query = '( "battery" OR "fuel cell" OR "supercapacitor" OR "electrolyzer" ) AND ' \
                       '( "impedance" OR "EIS" ) AND ' \
                       '( "Machine Learning" OR "Deep Learning" OR "Neural Network" OR "AI" OR "modeling" )'
        
        # Strategy 3: Electrochemistry-focused query
        electrochem_query = '( "electrochemistry" OR "electrochemical" ) AND ' \
                           '( "impedance" OR "EIS" ) AND ' \
                           '( "Machine Learning" OR "Deep Learning" OR "Neural Network" OR "AI" )'
        
        # Strategy 4: Corrosion-focused query
        corrosion_query = '( "corrosion" OR "coating" ) AND ' \
                         '( "impedance" OR "EIS" ) AND ' \
                         '( "Machine Learning" OR "Deep Learning" OR "Neural Network" OR "AI" )'
        
        # Strategy 5: Biosensor-focused query
        biosensor_query = '( "biosensor" OR "bioelectrochemistry" ) AND ' \
                         '( "impedance" OR "EIS" ) AND ' \
                         '( "Machine Learning" OR "Deep Learning" OR "Neural Network" OR "AI" )'
        
        queries = [broad_query, battery_query, electrochem_query, corrosion_query, biosensor_query]
        
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            update_log(f"Running query: {query[:100]}...")
            search = arxiv.Search(
                query=query,
                max_results=max_results * 5,  # Increased over-fetch
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in client.results(search):
                if not (start_year <= result.published.year <= end_year): 
                    continue
                
                # Check if any of the selected categories match
                if not any(cat in result.categories for cat in categories): 
                    continue
                
                paper_id = result.get_short_id()
                if paper_id in seen_ids: 
                    continue
                seen_ids.add(paper_id)
                
                rel_score = score_abstract_with_scibert(result.summary)
                if rel_score < 0.15:  # Lowered threshold
                    continue
                
                # Identify matched terms for the UI
                abstract_lower = result.summary.lower()
                found = [t for t in (EIS_CORE + AI_TERMS) if t.lower() in abstract_lower]

                all_papers.append({
                    "id": paper_id,
                    "title": result.title,
                    "authors": ", ".join([a.name for a in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary,
                    "pdf_url": result.pdf_url,
                    "matched_terms": ", ".join(found),
                    "relevance_prob": round(rel_score * 100, 2),
                    "query_type": query.split('(')[0].strip()
                })
        
        # Sort by relevance and return top results
        all_papers = sorted(all_papers, key=lambda x: x["relevance_prob"], reverse=True)
        return all_papers[:max_results]
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

# ==============================
# PDF & FILE HANDLERS
# ==============================
def handle_pdf_download(paper_id, pdf_url, paper_metadata):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, timeout=30)
        pdf_bytes = response.content
        st.session_state.downloaded_pdfs[paper_id] = pdf_bytes
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        doc.close()
        
        init_db(UNIVERSE_DB_FILE)
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        conn.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?)", 
                     (paper_id, paper_metadata['title'], paper_metadata['authors'], paper_metadata['year'], full_text))
        conn.commit()
        conn.close()
        st.session_state.universe_db_updated = True
        return True
    except Exception as e:
        update_log(f"PDF Error for {paper_id}: {e}")
        return False

# ==============================
# UI MAIN LAYOUT
# ==============================
with st.sidebar:
    st.header("⚙️ Configuration")
    max_res = st.slider("Max Results", 10, 1000, 200)  # Increased default and max
    
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        s_year = st.number_input("From", 2000, 2026, 2010)  # Extended range
    with col_y2:
        e_year = st.number_input("To", 2000, 2026, 2026)
        
    cats = st.multiselect("Categories", 
                          ["stat.ML", "cs.LG", "physics.chem-ph", "physics.app-ph", "electro-chem", 
                           "physics.comp-ph", "cs.AI", "cs.NE", "eess.SP"],  # Added more categories
                          default=["stat.ML", "cs.LG", "physics.chem-ph", "physics.app-ph"])
    
    search_triggered = st.button("🔍 Search EIS + AI", type="primary", use_container_width=True)

if search_triggered:
    st.session_state.search_performed = True
    with st.spinner("Analyzing scientific literature..."):
        results = query_arxiv_eis_ai(cats, max_res, s_year, e_year)
        if results:
            st.session_state.papers_df = pd.DataFrame(results)
            init_db(METADATA_DB_FILE)
            conn = sqlite3.connect(METADATA_DB_FILE)
            st.session_state.papers_df.to_sql("papers", conn, if_exists="replace", index=False)
            conn.close()
        else:
            st.warning("No papers matching the EIS + AI criteria found.")

# Display Results
if st.session_state.papers_df is not None:
    df = st.session_state.papers_df
    st.subheader(f"📚 Top {len(df)} Identified Papers")
    
    # Add filtering options
    st.sidebar.subheader("Filter Results")
    min_relevance = st.sidebar.slider("Minimum Relevance (%)", 0, 100, 15)
    filtered_df = df[df['relevance_prob'] >= min_relevance]
    
    # Add search box for title/abstract
    search_term = st.sidebar.text_input("Search in Title/Abstract")
    if search_term:
        mask = filtered_df['title'].str.contains(search_term, case=False) | filtered_df['abstract'].str.contains(search_term, case=False)
        filtered_df = filtered_df[mask]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} papers")
    
    for idx, row in filtered_df.iterrows():
        with st.expander(f"[{row['relevance_prob']}%] {row['title']} ({row['year']})"):
            st.markdown(f"**Authors:** {row['authors']}")
            if 'query_type' in row:
                st.markdown(f"**Found via:** {row['query_type']}")
            st.markdown(f"**Matched AI/EIS Concepts:** `{row['matched_terms']}`")
            st.write(row['abstract'])
            
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("📥 Index PDF", key=f"dl_{row['id']}"):
                    if handle_pdf_download(row['id'], row['pdf_url'], row.to_dict()):
                        st.success("Indexed!")
            with c2:
                st.markdown(f"[View on arXiv]({row['pdf_url'].replace('/pdf/', '/abs/')})")

    # Final Export Section
    st.markdown("---")
    st.subheader("💾 Export & Inspection")
    ec1, ec2, ec3 = st.columns(3)
    
    with ec1:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Export Metadata (CSV)", csv_data, "eis_ai_results.csv", "text/csv")
    
    with ec2:
        if os.path.exists(METADATA_DB_FILE):
            with open(METADATA_DB_FILE, "rb") as f:
                st.download_button("Download Metadata DB", f, "metadata.db")
                
    with ec3:
        if st.session_state.universe_db_updated:
            with open(UNIVERSE_DB_FILE, "rb") as f:
                st.download_button("Download Full-Text DB", f, "universe.db")

st.markdown("---")
st.text_area("📝 Logs", "\n".join(st.session_state.log_buffer), height=150)
