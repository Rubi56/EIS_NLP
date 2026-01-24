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

# Ensure the base directory exists
BASE_DIR.mkdir(exist_ok=True)

# Set up logging
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

st.title("🔋 EIS & AI Research Explorer")
st.markdown("""
This tool queries **arXiv** for the intersection of **Electrochemical Impedance Spectroscopy (EIS)** and **Artificial Intelligence**, focusing on:
- **EIS Analysis**: Nyquist/Bode plots, Equivalent Circuit Modeling (ECM), Distribution of Relaxation Times (DRT).
- **AI/ML Methods**: Machine Learning, Deep Learning (CNN/RNN/LSTM), Data-driven modeling, and NLP.
- **Applications**: Battery State of Health (SOH), Fuel Cells, Corrosion, and Biosensors.

It uses **SciBERT with attention-aware relevance scoring** and filters for papers where EIS concepts are mandatory.
""")

if IS_CLOUD:
    st.info("☁️ **Streamlit Cloud Mode**: Files are stored temporarily in `/tmp`. Download your databases before the session ends!")

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
    update_log("Loading SciBERT model (optimized for scientific text)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        update_log("SciBERT loaded successfully.")
        return tokenizer, model
    except Exception as e:
        update_log(f"Failed to load SciBERT: {e}")
        raise e

try:
    scibert_tokenizer, scibert_model = load_scibert()
except Exception as e:
    st.error(f"❌ Failed to load SciBERT: {e}.")
    st.stop()

# ==============================
# TEXT NORMALIZATION AND PATTERN DEFINITIONS
# ==============================
@st.cache_data
def normalize_text(text):
    text = text.lower()
    # Handle common EIS subscripts/shorthands
    shorthands = {'z\'': 'real impedance', 'z\'\'': 'imaginary impedance', 'ω': 'omega'}
    for key, val in shorthands.items():
        text = text.replace(key, val)
    return text

# MANDATORY TERMS (At least one must be present)
MANDATORY_EIS = ["eis", "impedance spectroscopy", "electrochemical impedance"]

# AI & DOMAIN KEYWORDS
KEY_TERMS = [
    "machine learning", "deep learning", "artificial intelligence", "neural networks",
    "data-driven", "natural language processing", "nlp", "large language models",
    "random forest", "gaussian process", "convolutional neural network", "cnn", "lstm",
    "equivalent circuit", "distribution of relaxation times", "drt", "nyquist", "bode plot",
    "state of health", "soh prediction", "battery aging", "transfer learning", 
    "physics-informed", "bayesian inference", "feature extraction"
]

# Optimized regex patterns for EIS + AI matching
KEY_PATTERNS = [
    r'\be?is\b|electrochemical impedance spectroscopy',
    r'\bmachine learning|deep learning|artificial intelligence|neural networks?\b',
    r'\bdata[- ]driven|predictive model(?:ing)?\b',
    r'\bnlp|natural language processing|transformer models?\b',
    r'\bcnn|lstm|rnn|gru|autoencoder\b',
    r'\beq?uivalent circuit (?:model|modeling|fitting)\b',
    r'\bnyquist|bode\s*plot\b',
    r'\bdrt|distribution of relaxation times\b',
    r'\bsoh|state\s*of\s*health|remaining useful life|rul\b',
    r'\bbattery|fuel cell|electrolyzer|corrosion|biosensor\b',
    r'\bphysics[- ]informed|pinn\b',
    r'\brand[o]m forest|xgboost|support vector machine|svm\b'
]

@st.cache_data
def compile_patterns():
    return [re.compile(pattern, re.IGNORECASE) for pattern in KEY_PATTERNS]

COMPILED_PATTERNS = compile_patterns()

# ==============================
# SCIBERT-BASED ABSTRACT SCORING
# ==============================
@st.cache_data
def score_abstract_with_scibert(abstract):
    try:
        abstract_norm = normalize_text(abstract)
        
        # STRICT REQUIREMENT: Must mention EIS/Impedance
        if not any(term in abstract_norm for term in MANDATORY_EIS):
            return 0.0

        inputs = scibert_tokenizer(
            abstract, return_tensors="pt", truncation=True, 
            max_length=512, padding=True, return_attention_mask=True
        )
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        
        # Regex scoring
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_norm))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        
        # Attention boost for AI + EIS specific tokens
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        ai_eis_tokens = ['impedance', 'eis', 'learn', 'network', 'data', 'circuit', 'predict']
        keyword_indices = [i for i, t in enumerate(tokens) if any(kw in t.lower() for kw in ai_eis_tokens)]
        
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.05:
                boost = 0.25 * (len(keyword_indices) / len(tokens))
                relevance_prob = min(relevance_prob + boost, 1.0)
        
        return relevance_prob
    except Exception as e:
        update_log(f"Scoring failed: {e}")
        return 0.1

# ==============================
# DATABASE LOGIC
# ==============================
def init_db(db_path, is_universe=False):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if is_universe:
        cursor.execute("CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, content TEXT)")
    else:
        cursor.execute("CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, categories TEXT, abstract TEXT, pdf_url TEXT, matched_terms TEXT, relevance_prob REAL)")
    conn.commit()
    conn.close()

# ==============================
# ARXIV QUERYING
# ==============================
@st.cache_data
def query_arxiv_api(query, categories, max_results, start_year, end_year):
    try:
        client = arxiv.Client()
        # We search for EIS broadly, then filter for AI in the scoring
        search = arxiv.Search(
            query=query,
            max_results=max_results * 3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        seen_ids = set()
        
        for result in client.results(search):
            paper_id = result.get_short_id()
            if paper_id in seen_ids or not (start_year <= result.published.year <= end_year):
                continue
            if not any(cat in result.categories for cat in categories):
                continue
            
            relevance_prob = score_abstract_with_scibert(result.summary)
            if relevance_prob < 0.3: # User requested 30% threshold
                continue
                
            seen_ids.add(paper_id)
            
            # Highlight AI and EIS terms
            abstract_highlighted = result.summary
            for term in (MANDATORY_EIS + KEY_TERMS[:10]):
                abstract_highlighted = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    f'<span style="background-color: #D1ECF1; color: #0C5460; font-weight: bold;">{term}</span>',
                    abstract_highlighted, flags=re.IGNORECASE
                )

            papers.append({
                "id": paper_id,
                "title": result.title,
                "authors": ", ".join([a.name for a in result.authors]),
                "year": result.published.year,
                "categories": ", ".join(result.categories),
                "abstract": result.summary,
                "abstract_highlighted": abstract_highlighted,
                "pdf_url": result.pdf_url,
                "matched_terms": ", ".join([t for t in (MANDATORY_EIS + KEY_TERMS) if t.lower() in result.summary.lower()]),
                "relevance_prob": round(relevance_prob * 100, 2)
            })
            if len(papers) >= max_results: break
            
        return sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# ==============================
# PDF & FILE HANDLERS
# ==============================
def handle_pdf_download(paper_id, pdf_url, paper_metadata):
    try:
        response = requests.get(pdf_url, timeout=30)
        pdf_bytes = response.content
        st.session_state.downloaded_pdfs[paper_id] = pdf_bytes
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        doc.close()
        
        init_db(UNIVERSE_DB_FILE, is_universe=True)
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        conn.execute("INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?)", 
                     (paper_id, paper_metadata['title'], paper_metadata['authors'], paper_metadata['year'], full_text))
        conn.commit()
        conn.close()
        st.session_state.universe_db_updated = True
        return True
    except Exception as e:
        update_log(f"PDF Error: {e}")
        return False

# ==============================
# MAIN UI
# ==============================
with st.sidebar:
    st.header("⚙️ EIS + AI Config")
    # Enforce EIS in the query
    base_query = '( "Electrochemical Impedance Spectroscopy" OR "EIS" )'
    ai_topic = st.selectbox("Focus AI Area", ["Machine Learning", "Deep Learning", "NLP", "Data-driven"])
    
    full_query = f'{base_query} AND ("{ai_topic}" OR "Neural Network")'
    
    st.caption("Constructed API Query:")
    st.code(full_query)

    categories = st.multiselect("Categories", ["stat.ML", "cs.LG", "physics.chem-ph", "physics.app-ph", "math.ST"], default=["stat.ML", "physics.chem-ph"])
    max_results = st.slider("Max Results", 10, 100, 30)
    start_year = st.number_input("Start", 2015, 2026, 2018)
    
    search_button = st.button("🔎 Search Papers", type="primary")

if search_button:
    st.session_state.search_performed = True
    with st.spinner("Analyzing arXiv literature..."):
        results = query_arxiv_api(full_query, categories, max_results, start_year, 2026)
        if results:
            st.session_state.papers_df = pd.DataFrame(results)
            init_db(METADATA_DB_FILE)
            conn = sqlite3.connect(METADATA_DB_FILE)
            st.session_state.papers_df.drop(columns=["abstract_highlighted"]).to_sql("papers", conn, if_exists="replace", index=False)
            conn.close()
        else:
            st.warning("No papers found matching the 30% relevance criteria for EIS + AI.")

# DISPLAY RESULTS
if st.session_state.search_performed and st.session_state.papers_df is not None:
    st.subheader(f"📑 Top Research Papers (Relevance > 30%)")
    for _, paper in st.session_state.papers_df.iterrows():
        with st.expander(f"[{paper['relevance_prob']}%] {paper['title']}"):
            st.write(f"**Authors:** {paper['authors']} | **Year:** {paper['year']}")
            st.markdown(paper['abstract_highlighted'], unsafe_allow_html=True)
            if st.button("📥 Index Full Text", key=f"dl_{paper['id']}"):
                if handle_pdf_download(paper['id'], paper['pdf_url'], paper.to_dict()):
                    st.success("Paper added to Universe DB!")

    # EXPORTS
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("💾 Download Metadata DB", open(METADATA_DB_FILE, "rb") if METADATA_DB_FILE.exists() else b"", "eis_metadata.db")
    with c2:
        if st.session_state.universe_db_updated:
            st.download_button("🧠 Download Universe DB", open(UNIVERSE_DB_FILE, "rb"), "eis_universe.db")
    with c3:
        csv = st.session_state.papers_df.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Export CSV", csv, "eis_ai_results.csv", "text/csv")

st.divider()
st.text_area("📝 Logs", "\n".join(st.session_state.log_buffer), height=150)
