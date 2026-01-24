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
    page_title="EIS and AI Knowledge Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Electrochemical Impedance Spectroscopy (EIS) with AI/ML Knowledge Explorer")
st.markdown("""
This advanced tool queries **arXiv** for scientific literature on **Electrochemical Impedance Spectroscopy (EIS)** combined with **Artificial Intelligence/Machine Learning** methods, with focus on:
- **EIS fundamentals** (Nyquist, Bode plots, equivalent circuits)
- **AI/ML applications** (neural networks, deep learning, data-driven models)
- **Natural Language Processing** for EIS data interpretation
- **Battery/Supercapacitor/Fuel cell** applications
- **Corrosion monitoring** and **biomedical sensing**
- **Automated equivalent circuit fitting**
- **EIS data augmentation** and **synthetic data generation**
- **Transfer learning** for EIS analysis

It uses **SciBERT with attention-aware relevance scoring** (>30% threshold) and stores:
- **Metadata** in `eis_ai_metadata.db`
- **Full extracted text** in `eis_ai_universe.db`

All data is downloadable. **No automated PDF scraping** — downloads occur only on user request.
""")

if IS_CLOUD:
    st.info("☁️ **Streamlit Cloud Mode**: All files are stored temporarily in `/tmp`. Use download buttons before your session expires!")

# ==============================
# SESSION STATE INITIALIZATION
# ==============================
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "downloaded_pdfs" not in st.session_state:
    st.session_state.downloaded_pdfs = {}  # {paper_id: pdf_bytes}
if "universe_db_updated" not in st.session_state:
    st.session_state.universe_db_updated = False
if "papers_df" not in st.session_state:
    st.session_state.papers_df = None
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

def update_log(message):
    """Add a timestamped message to the log buffer and file."""
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
    """Load and cache the SciBERT tokenizer and model."""
    update_log("Loading SciBERT model and tokenizer from Hugging Face...")
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
    st.error(f"❌ Failed to load SciBERT: {e}. Please install: `pip install transformers torch`")
    st.stop()

# ==============================
# TEXT NORMALIZATION AND PATTERN DEFINITIONS
# ==============================
@st.cache_data
def normalize_text(text):
    """Normalize text by replacing Greek letters, subscripts, and superscripts."""
    # Greek letters
    greek_to_latin = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
        'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
        'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
        'Α': 'alpha', 'Β': 'beta', 'Γ': 'gamma', 'Δ': 'delta', 'Ε': 'epsilon',
        'Φ': 'phi', 'Ω': 'omega'
    }
    for greek, latin in greek_to_latin.items():
        text = text.replace(greek, latin)
    
    # Subscripts
    subscripts = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')'
    }
    for sub, digit in subscripts.items():
        text = text.replace(sub, digit)
    
    # Superscripts
    superscripts = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')'
    }
    for sup, digit in superscripts.items():
        text = text.replace(sup, digit)
    
    return text.lower()

# Comprehensive list of key terms for EIS and AI
KEY_TERMS = [
    # Mandatory EIS terms
    "electrochemical impedance spectroscopy", "EIS", "impedance spectroscopy",
    "Nyquist plot", "Bode plot", "complex impedance", "Z''", "Z'",
    "equivalent circuit", "Randles circuit", "Warburg impedance",
    "charge transfer resistance", "double layer capacitance", "solution resistance",
    "constant phase element", "CPE", "Faradaic impedance", "non-Faradaic impedance",
    
    # AI/ML terms (comprehensive)
    "artificial intelligence", "AI", "machine learning", "ML", "deep learning",
    "neural network", "convolutional neural network", "CNN", "recurrent neural network", "RNN",
    "LSTM", "GRU", "transformer", "attention mechanism", "self-attention",
    "natural language processing", "NLP", "large language model", "LLM",
    "GPT", "BERT", "transformer model", "language model",
    "data-driven", "data-driven model", "data-driven approach",
    "supervised learning", "unsupervised learning", "semi-supervised learning",
    "reinforcement learning", "transfer learning", "few-shot learning",
    "regression", "classification", "clustering", "dimensionality reduction",
    "PCA", "t-SNE", "autoencoder", "variational autoencoder", "VAE",
    "generative adversarial network", "GAN", "diffusion model",
    "random forest", "support vector machine", "SVM", "gradient boosting",
    "XGBoost", "LightGBM", "CatBoost", "ensemble learning",
    
    # AI applications in EIS
    "automated equivalent circuit fitting", "EIS data analysis",
    "EIS prediction", "EIS classification", "EIS regression",
    "impedance data augmentation", "synthetic EIS data",
    "EIS feature extraction", "impedance feature engineering",
    "EIS anomaly detection", "impedance fault diagnosis",
    "real-time EIS monitoring", "online EIS analysis",
    
    # Materials and applications
    "battery", "lithium-ion battery", "LIB", "solid-state battery",
    "supercapacitor", "fuel cell", "PEMFC", "SOFC",
    "corrosion", "corrosion monitoring", "corrosion detection",
    "biomedical", "biosensor", "impedance biosensor",
    "electrocatalyst", "electrode", "electrolyte",
    "sensor", "impedance sensor", "electrochemical sensor",
    
    # EIS parameters and analysis
    "impedance modulus", "phase angle", "frequency response",
    "Kramers-Kronig relations", "distribution of relaxation times", "DRT",
    "electrochemical kinetics", "charge transfer kinetics",
    "mass transport", "diffusion coefficient",
    "electrode-electrolyte interface", "solid electrolyte interface", "SEI",
    
    # Advanced AI/ML techniques
    "bayesian optimization", "gaussian process", "kernel method",
    "feature selection", "hyperparameter tuning", "model optimization",
    "explainable AI", "XAI", "model interpretation", "SHAP", "LIME",
    "time series analysis", "spectral analysis", "frequency domain analysis",
    "signal processing", "digital signal processing", "DSP",
    
    # Data science and processing
    "big data", "data mining", "pattern recognition",
    "computer vision", "image processing", "spectrum processing",
    "data fusion", "multimodal data", "multi-source data",
    "cloud computing", "edge computing", "IoT", "internet of things",
    
    # Software and tools
    "Python", "TensorFlow", "PyTorch", "Keras", "scikit-learn",
    "ZView", "ZPlot", "EC-Lab", "Gamry", "BioLogic",
    "MATLAB", "Simulink", "LabVIEW",
    
    # Related techniques
    "cyclic voltammetry", "CV", "chronoamperometry", "chronopotentiometry",
    "electrochemical noise", "potentiostatic", "galvanostatic"
]

# Optimized regex patterns for fast matching
KEY_PATTERNS = [
    # Mandatory EIS patterns
    r'\belectrochemical impedance spectroscopy|EIS|impedance spectroscopy\b',
    r'\bnyquist plot|bode plot|complex impedance\b',
    r'\bequivalent circuit|randles circuit|warburg impedance\b',
    r'\bcharge transfer resistance|double layer capacitance|solution resistance\b',
    r'\bconstant phase element|CPE|faradaic impedance\b',
    
    # AI/ML mandatory patterns
    r'\bartificial intelligence|AI\b',
    r'\bmachine learning|ML\b',
    r'\bdeep learning|neural network\b',
    r'\bnatural language processing|NLP\b',
    r'\blarge language model|LLM\b',
    r'\bdata-driven|data driven\b',
    
    # AI techniques
    r'\bconvolutional neural network|CNN\b',
    r'\brecurrent neural network|RNN|LSTM|GRU\b',
    r'\btransformer|attention mechanism|self-attention\b',
    r'\bGPT|BERT|transformer model|language model\b',
    r'\bsupervised learning|unsupervised learning|semi-supervised learning\b',
    r'\breinforcement learning|transfer learning|few-shot learning\b',
    
    # Applications in EIS
    r'\bautomated equivalent circuit fitting\b',
    r'\bEIS data analysis|impedance data analysis\b',
    r'\bEIS prediction|impedance prediction\b',
    r'\bEIS classification|impedance classification\b',
    r'\bEIS regression|impedance regression\b',
    r'\bEIS data augmentation|synthetic EIS data\b',
    r'\bEIS feature extraction|impedance feature extraction\b',
    
    # Materials and systems
    r'\bbattery|lithium-ion battery|LIB\b',
    r'\bsupercapacitor|fuel cell\b',
    r'\bcorrosion monitoring|corrosion detection\b',
    r'\bbiomedical|biosensor|impedance biosensor\b',
    
    # EIS analysis techniques
    r'\bKramers-Kronig relations|distribution of relaxation times|DRT\b',
    r'\belectrochemical kinetics|charge transfer kinetics\b',
    r'\bmass transport|diffusion coefficient\b',
    r'\belectrode-electrolyte interface|solid electrolyte interface|SEI\b',
    
    # Advanced AI techniques
    r'\bbayesian optimization|gaussian process|kernel method\b',
    r'\bfeature selection|hyperparameter tuning\b',
    r'\bexplainable AI|XAI|model interpretation|SHAP|LIME\b',
    r'\btime series analysis|spectral analysis\b',
    
    # Software and implementation
    r'\bPython|TensorFlow|PyTorch|Keras|scikit-learn\b',
    r'\bZView|ZPlot|EC-Lab|Gamry|BioLogic\b',
    r'\bMATLAB|Simulink|LabVIEW\b'
]

@st.cache_data
def compile_patterns():
    """Compile regex patterns for efficiency."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in KEY_PATTERNS]

COMPILED_PATTERNS = compile_patterns()

# ==============================
# SCIBERT-BASED ABSTRACT SCORING WITH ATTENTION BOOST
# ==============================
@st.cache_data
def score_abstract_with_scibert(abstract):
    """Score abstract relevance using SciBERT and regex pattern matching."""
    try:
        # Tokenize and run SciBERT
        inputs = scibert_tokenizer(
            abstract,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        
        # Regex-based relevance (lenient with sqrt)
        abstract_normalized = normalize_text(abstract)
        
        # Check for mandatory terms: EIS and at least one AI term
        has_eis = any(pat.search(abstract_normalized) for pat in COMPILED_PATTERNS[:5])
        has_ai = any(pat.search(abstract_normalized) for pat in COMPILED_PATTERNS[5:11])
        
        if not (has_eis and has_ai):
            # Paper doesn't contain both EIS and AI terms
            update_log(f"Paper missing mandatory terms (EIS: {has_eis}, AI: {has_ai})")
            return 0.0
        
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        
        # Attention-based boost for key terms
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [
            i for i, token in enumerate(tokens)
            if any(kw in token.lower() for kw in ['eis', 'impedance', 'machine', 'learning', 'neural', 'network', 'ai', 'data', 'model'])
        ]
        if keyword_indices:
            # Use last layer, first attention head
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1:
                boost = 0.3 * (len(keyword_indices) / len(tokens))
                relevance_prob = min(relevance_prob + boost, 1.0)
        
        update_log(f"SciBERT (attention-boosted) scored abstract: {relevance_prob:.3f} (patterns matched: {num_matched})")
        return relevance_prob
    except Exception as e:
        update_log(f"SciBERT scoring failed: {str(e)}")
        # Fallback to regex-only scoring
        abstract_normalized = normalize_text(abstract)
        
        # Check for mandatory terms
        has_eis = any(pat.search(abstract_normalized) for pat in COMPILED_PATTERNS[:5])
        has_ai = any(pat.search(abstract_normalized) for pat in COMPILED_PATTERNS[5:11])
        
        if not (has_eis and has_ai):
            return 0.0
            
        num_matched = sum(1 for pat in COMPILED_PATTERNS if pat.search(abstract_normalized))
        relevance_prob = np.sqrt(num_matched) / np.sqrt(len(KEY_PATTERNS))
        update_log(f"Fallback scoring: {relevance_prob:.3f}")
        return relevance_prob

# ==============================
# DATABASE INITIALIZATION FUNCTIONS
# ==============================
def init_metadata_db():
    """Initialize the metadata SQLite database."""
    try:
        conn = sqlite3.connect(METADATA_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                matched_terms TEXT,
                relevance_prob REAL,
                has_eis BOOLEAN,
                has_ai BOOLEAN
            )
        """)
        conn.commit()
        conn.close()
        update_log(f"Initialized metadata database at {METADATA_DB_FILE}")
    except Exception as e:
        update_log(f"Failed to initialize metadata DB: {e}")
        st.error(f"Database error: {e}")

def init_universe_db():
    """Initialize the full-text universe SQLite database."""
    try:
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT,
                eis_sections TEXT,
                ai_sections TEXT
            )
        """)
        conn.commit()
        conn.close()
        update_log(f"Initialized universe database at {UNIVERSE_DB_FILE}")
    except Exception as e:
        update_log(f"Failed to initialize universe DB: {e}")
        st.error(f"Database error: {e}")

# ==============================
# ARXIV QUERY FUNCTION WITH DUPLICATE PREVENTION
# ==============================
@st.cache_data
def query_arxiv_api(query, categories, max_results, start_year, end_year):
    """Query arXiv and return relevant papers with uniqueness guarantee."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results * 3,  # Fetch extra to account for filtering
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        query_terms = [t.strip() for t in query.split(' OR ')]
        query_words = {t.strip('"').lower() for t in query_terms}
        seen_ids = set()  # Prevent duplicate paper IDs
        
        for result in client.results(search):
            # Year filtering (start from 2010 as requested)
            if not (start_year <= result.published.year <= end_year):
                continue
            if not any(cat in result.categories for cat in categories):
                continue
            
            # Ensure uniqueness
            paper_id = result.get_short_id()
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            
            # Check for mandatory terms: EIS and AI
            abstract_lower = result.summary.lower()
            title_lower = result.title.lower()
            
            # Check for EIS terms
            eis_terms = ['electrochemical impedance spectroscopy', 'eis', 'impedance spectroscopy']
            has_eis = any(term in abstract_lower or term in title_lower for term in eis_terms)
            
            # Check for AI/ML terms
            ai_terms = ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 
                       'natural language processing', 'data-driven', 'ai', 'ml']
            has_ai = any(term in abstract_lower or term in title_lower for term in ai_terms)
            
            # Only include papers that have BOTH EIS and AI terms
            if not (has_eis and has_ai):
                continue
            
            # Match other terms in title or abstract
            matched_terms = [term for term in query_words if term in abstract_lower or term in title_lower]
            if not matched_terms:
                continue
            
            # Score relevance
            relevance_prob = score_abstract_with_scibert(result.summary)
            
            # Skip papers below threshold
            if relevance_prob < 0.3:  # 30% threshold
                continue
            
            # Highlight matched terms in abstract
            abstract_highlighted = result.summary
            for term in matched_terms:
                abstract_highlighted = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    f'<span style="background-color: #FFF3CD; color: #856404; font-weight: bold;">{term}</span>',
                    abstract_highlighted,
                    flags=re.IGNORECASE
                )
            
            papers.append({
                "id": paper_id,
                "title": result.title,
                "authors": ", ".join([author.name for author in result.authors]),
                "year": result.published.year,
                "categories": ", ".join(result.categories),
                "abstract": result.summary,
                "abstract_highlighted": abstract_highlighted,
                "pdf_url": result.pdf_url,
                "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                "relevance_prob": round(relevance_prob * 100, 2),
                "has_eis": has_eis,
                "has_ai": has_ai
            })
            
            if len(papers) >= max_results:
                break
        
        # Sort by relevance
        papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
        update_log(f"Query returned {len(papers)} unique papers with EIS+AI")
        return papers
    except Exception as e:
        update_log(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}. Try simplifying the query.")
        return []

# ==============================
# PDF DOWNLOAD AND FULL-TEXT EXTRACTION
# ==============================
def download_pdf_bytes(pdf_url):
    """Download a PDF as bytes with proper headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; EIS-AI Research Tool/1.0; +https://github.com/your-repo)'
    }
    response = requests.get(pdf_url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content

def extract_sections_from_text(full_text):
    """Extract EIS and AI related sections from text."""
    sections = {
        "eis_sections": [],
        "ai_sections": []
    }
    
    # Normalize text
    text_lower = full_text.lower()
    
    # Find EIS-related sections
    eis_keywords = ['eis', 'electrochemical impedance', 'impedance spectroscopy', 
                   'nyquist', 'bode', 'equivalent circuit']
    for keyword in eis_keywords:
        if keyword in text_lower:
            # Find context around keyword
            idx = text_lower.find(keyword)
            start = max(0, idx - 200)
            end = min(len(text_lower), idx + 200)
            sections["eis_sections"].append(f"...{full_text[start:end]}...")
    
    # Find AI-related sections
    ai_keywords = ['machine learning', 'deep learning', 'neural network', 'artificial intelligence',
                  'data-driven', 'nlp', 'natural language processing', 'ai', 'ml']
    for keyword in ai_keywords:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            start = max(0, idx - 200)
            end = min(len(text_lower), idx + 200)
            sections["ai_sections"].append(f"...{full_text[start:end]}...")
    
    # Limit to top 3 sections each
    sections["eis_sections"] = sections["eis_sections"][:3]
    sections["ai_sections"] = sections["ai_sections"][:3]
    
    return sections

def handle_pdf_download(paper_id, pdf_url, paper_metadata):
    """Download a PDF, extract text, and update databases."""
    try:
        # Download PDF
        pdf_bytes = download_pdf_bytes(pdf_url)
        st.session_state.downloaded_pdfs[paper_id] = pdf_bytes
        update_log(f"Downloaded PDF for {paper_id} ({len(pdf_bytes)} bytes)")
        
        # Extract full text
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page_num, page in enumerate(doc):
            full_text += page.get_text()
        doc.close()
        update_log(f"Extracted {len(full_text)} characters from {paper_id}")
        
        # Extract EIS and AI sections
        sections = extract_sections_from_text(full_text)
        
        # Initialize and update universe DB
        init_universe_db()
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content, eis_sections, ai_sections)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_id,
            paper_metadata.get("title", ""),
            paper_metadata.get("authors", "Unknown"),
            paper_metadata.get("year", 0),
            full_text,
            " ||| ".join(sections["eis_sections"]),
            " ||| ".join(sections["ai_sections"])
        ))
        conn.commit()
        conn.close()
        st.session_state.universe_db_updated = True
        update_log(f"Updated universe DB with {paper_id}")
        return True
    except Exception as e:
        error_msg = f"PDF download/extraction failed for {paper_id}: {str(e)}"
        update_log(error_msg)
        st.error(error_msg)
        return False

# ==============================
# FILE CREATION UTILITIES
# ==============================
def create_zip_of_downloaded_pdfs():
    """Create a ZIP file of all downloaded PDFs."""
    if not st.session_state.downloaded_pdfs:
        return None
    zip_path = BASE_DIR / "eis_ai_pdfs.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for paper_id, pdf_bytes in st.session_state.downloaded_pdfs.items():
            zipf.writestr(f"{paper_id}.pdf", pdf_bytes)
    update_log(f"Created ZIP with {len(st.session_state.downloaded_pdfs)} PDFs")
    return zip_path

def get_db_as_bytes(db_path):
    """Read a SQLite database file as bytes."""
    if not db_path.exists():
        return None
    with open(db_path, "rb") as f:
        return f.read()

# ==============================
# DATABASE INSPECTION FUNCTIONS
# ==============================
def inspect_metadata_db():
    """Display the metadata database contents."""
    if not METADATA_DB_FILE.exists():
        st.warning("_Metadata database not found._")
        return
    
    conn = sqlite3.connect(METADATA_DB_FILE)
    df = pd.read_sql("SELECT * FROM papers ORDER BY relevance_prob DESC", conn)
    conn.close()
    
    st.subheader("🗃️ Metadata Database Inspection")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", len(df))
    with col2:
        st.metric("Avg Relevance", f"{df['relevance_prob'].mean():.1f}%")
    with col3:
        st.metric("Year Range", f"{df['year'].min()}-{df['year'].max()}")
    with col4:
        st.metric("Top Category", df['categories'].str.split(',').explode().mode().iloc[0] if not df.empty else "N/A")
    
    # Allow download
    metadata_bytes = get_db_as_bytes(METADATA_DB_FILE)
    if metadata_bytes:
        st.download_button(
            "📥 Download Metadata DB",
            metadata_bytes,
            file_name="eis_ai_metadata.db",
            mime="application/x-sqlite3"
        )

def inspect_universe_db():
    """Display the universe database contents with search."""
    if not st.session_state.universe_db_updated or not UNIVERSE_DB_FILE.exists():
        st.warning("_Full-text database not available. Download at least one PDF first._")
        return
    
    conn = sqlite3.connect(UNIVERSE_DB_FILE)
    df = pd.read_sql("SELECT id, title, authors, year, eis_sections, ai_sections FROM papers", conn)
    conn.close()
    
    st.subheader("🔍 Full-Text (Universe) Database Inspection")
    
    # Search box
    search_term = st.text_input("Search full text for:", key="universe_search")
    
    if search_term:
        conn = sqlite3.connect(UNIVERSE_DB_FILE)
        query = """
        SELECT id, title, authors, year, 
               substr(content, instr(lower(content), lower(?)) - 100, 200) as snippet
        FROM papers 
        WHERE lower(content) LIKE ?
        """
        df_results = pd.read_sql_query(query, conn, params=(search_term, f"%{search_term.lower()}%"))
        conn.close()
        
        if not df_results.empty:
            st.write(f"Found {len(df_results)} results for '{search_term}':")
            for _, row in df_results.iterrows():
                with st.expander(f"📄 {row['title']} ({row['year']})"):
                    st.markdown(f"**Authors**: {row['authors']}")
                    st.markdown(f"**Snippet**: ...{row['snippet']}...")
        else:
            st.info("No matches found.")
    
    # Show paper list with EIS/AI sections
    st.markdown("### All Indexed Papers with EIS/AI Sections")
    
    for idx, row in df.iterrows():
        with st.expander(f"📊 {row['title']} ({row['year']})"):
            st.markdown(f"**Authors**: {row['authors']}")
            
            # Display EIS sections
            if row['eis_sections'] and row['eis_sections'] != '':
                st.markdown("**EIS-Related Sections:**")
                sections = row['eis_sections'].split(' ||| ')
                for i, section in enumerate(sections[:2], 1):
                    st.markdown(f"{i}. {section}")
            
            # Display AI sections
            if row['ai_sections'] and row['ai_sections'] != '':
                st.markdown("**AI/ML-Related Sections:**")
                sections = row['ai_sections'].split(' ||| ')
                for i, section in enumerate(sections[:2], 1):
                    st.markdown(f"{i}. {section}")
    
    # Download button
    universe_bytes = get_db_as_bytes(UNIVERSE_DB_FILE)
    if universe_bytes:
        st.download_button(
            "📥 Download Full-Text DB",
            universe_bytes,
            file_name="eis_ai_universe.db",
            mime="application/x-sqlite3"
        )

# ==============================
# MAIN APPLICATION LAYOUT
# ==============================
st.header("🔍 arXiv Query for EIS with AI/ML Applications")
st.markdown("""
Use the sidebar to configure your search. Results are filtered to include **only papers containing BOTH EIS and AI/ML terms**.
Papers are scored by **SciBERT + regex relevance** (>30% threshold).
""")

# Log display
log_container = st.empty()
def display_logs():
    log_container.text_area("📝 Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

# Sidebar controls
with st.sidebar:
    st.image("https://arxiv.org/favicon.ico", width=32)
    st.subheader("⚙️ Search Configuration")
    
    # Query construction
    query_mode = st.radio("Query Mode", ["Auto (Recommended)", "Custom"], horizontal=True)
    if query_mode == "Auto":
        # Build query that ensures both EIS and AI terms
        eis_terms = ['"electrochemical impedance spectroscopy"', '"EIS"', '"impedance spectroscopy"']
        ai_terms = ['"artificial intelligence"', '"machine learning"', '"deep learning"', 
                   '"neural network"', '"natural language processing"', '"data-driven"']
        
        # Create query that requires both EIS AND AI
        query = f'({" OR ".join(eis_terms)}) AND ({" OR ".join(ai_terms)})'
        st.text_area("Auto-generated Query", value=query, height=100, disabled=True)
    else:
        query = st.text_area("Custom Query", 
                           value='("electrochemical impedance spectroscopy" OR "EIS") AND ("machine learning" OR "artificial intelligence")',
                           height=100)
    
    # Categories
    default_categories = ["physics.chem-ph", "cond-mat.mtrl-sci", "cs.LG", "cs.AI", "physics.app-ph"]
    categories = st.multiselect(
        "arXiv Categories",
        options=default_categories + ["cs.CL", "cs.CV", "stat.ML", "physics.ins-det", "physics.comp-ph"],
        default=default_categories
    )
    
    # Limits
    max_results = st.slider("Max Results", min_value=1, max_value=200, value=50)
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=2010, max_value=current_year, value=2010)
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
    
    # Filter options
    st.markdown("---")
    st.subheader("🔍 Advanced Filters")
    min_relevance = st.slider("Minimum Relevance (%)", min_value=30, max_value=100, value=30)
    
    # Action button
    st.markdown("---")
    search_button = st.button("🚀 Execute Search", type="primary", use_container_width=True)

# Main content area
if search_button:
    if not categories:
        st.error("⚠️ Please select at least one arXiv category.")
    elif start_year > end_year:
        st.error("⚠️ Start year cannot be greater than end year.")
    else:
        st.session_state.search_performed = True
        with st.spinner("📡 Querying arXiv API (this may take a minute)..."):
            papers = query_arxiv_api(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("📭 No papers found. Try broadening your query or categories.")
        else:
            st.success(f"✅ Found **{len(papers)}** relevant papers (relevance > {min_relevance}%).")
            
            # Filter by relevance threshold
            relevant_papers = [p for p in papers if p["relevance_prob"] > min_relevance]
            if not relevant_papers:
                st.warning(f"📭 No papers above {min_relevance}% relevance threshold.")
            else:
                df = pd.DataFrame(relevant_papers)
                st.session_state.papers_df = df
                
                # Save to metadata DB
                init_metadata_db()
                conn = sqlite3.connect(METADATA_DB_FILE)
                df.drop(columns=["abstract_highlighted"]).to_sql("papers", conn, if_exists="replace", index=False)
                conn.close()
                update_log(f"Saved {len(df)} papers to metadata DB")
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Papers Found", len(df))
                with col2:
                    avg_rel = df['relevance_prob'].mean()
                    st.metric("Average Relevance", f"{avg_rel:.1f}%")
                with col3:
                    year_range = f"{df['year'].min()}-{df['year'].max()}"
                    st.metric("Year Range", year_range)
                
                # Display papers
                st.subheader("📚 Relevant Papers")
                for idx, paper in df.iterrows():
                    with st.expander(f"📄 **{paper['title']}** ({paper['year']}) — **{paper['relevance_prob']}%**"):
                        st.markdown(f"**Authors**: {paper['authors']}")
                        st.markdown(f"**Categories**: `{paper['categories']}`")
                        st.markdown(f"**Matched Terms**: `{paper['matched_terms']}`")
                        
                        # EIS and AI indicators
                        col_eis, col_ai = st.columns(2)
                        with col_eis:
                            st.success("✅ Contains EIS") if paper['has_eis'] else st.error("❌ No EIS")
                        with col_ai:
                            st.success("✅ Contains AI/ML") if paper['has_ai'] else st.error("❌ No AI/ML")
                        
                        st.markdown("### Abstract")
                        st.markdown(paper["abstract_highlighted"], unsafe_allow_html=True)
                        
                        col_btn, col_links = st.columns([1, 3])
                        with col_btn:
                            # Unique key using paper ID
                            if st.button("📥 Download PDF", key=f"download_{paper['id']}"):
                                with st.spinner("Downloading and extracting text..."):
                                    success = handle_pdf_download(paper["id"], paper["pdf_url"], paper.to_dict())
                                    if success:
                                        st.success("✅ PDF downloaded and indexed!")
                        with col_links:
                            abs_url = paper['pdf_url'].replace('/pdf/', '/abs/')
                            st.markdown(f"[🌐 View on arXiv]({abs_url}) | [📄 Direct PDF]({paper['pdf_url']})")

# Always show download and inspection section if search was performed
if st.session_state.search_performed:
    st.markdown("---")
    st.subheader("📥 Download & Inspect Results")
    
    # File downloads
    col1, col2, col3 = st.columns(3)
    
    # PDF ZIP
    with col1:
        if st.session_state.downloaded_pdfs:
            zip_path = create_zip_of_downloaded_pdfs()
            if zip_path:
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "📦 All PDFs (ZIP)",
                        f,
                        file_name="eis_ai_pdfs.zip",
                        mime="application/zip"
                    )
        else:
            st.button("📦 All PDFs (ZIP)", disabled=True, help="Download at least one PDF first")
    
    # Metadata DB
    with col2:
        metadata_bytes = get_db_as_bytes(METADATA_DB_FILE)
        if metadata_bytes:
            st.download_button(
                "🗃️ Metadata DB",
                metadata_bytes,
                file_name="eis_ai_metadata.db",
                mime="application/x-sqlite3"
            )
        else:
            st.button("🗃️ Metadata DB", disabled=True, help="Search first")
    
    # Universe DB
    with col3:
        universe_bytes = get_db_as_bytes(UNIVERSE_DB_FILE)
        if universe_bytes:
            st.download_button(
                "🔍 Full-Text DB",
                universe_bytes,
                file_name="eis_ai_universe.db",
                mime="application/x-sqlite3"
            )
        else:
            st.button("🔍 Full-Text DB", disabled=True, help="Download at least one PDF first")
    
    # CSV/JSON exports
    if st.session_state.papers_df is not None:
        col4, col5 = st.columns(2)
        with col4:
            csv = st.session_state.papers_df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
            st.download_button("📋 Export as CSV", csv, "eis_ai_papers.csv", "text/csv")
        with col5:
            json_data = st.session_state.papers_df.drop(columns=["abstract_highlighted"]).to_json(orient="records", indent=2)
            st.download_button("🧾 Export as JSON", json_data, "eis_ai_papers.json", "application/json")
    
    # Database inspection tabs
    st.markdown("### 🔍 Database Inspection")
    tab1, tab2 = st.tabs(["Metadata DB", "Full-Text (Universe) DB"])
    with tab1:
        inspect_metadata_db()
    with tab2:
        inspect_universe_db()

# Always show logs at bottom
st.markdown("---")
display_logs()

# Add information section
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    ### EIS + AI/ML Knowledge Explorer
    
    **Purpose**: This tool systematically searches for research papers that combine:
    1. **Electrochemical Impedance Spectroscopy (EIS)** - A powerful electrochemical technique
    2. **Artificial Intelligence/Machine Learning (AI/ML)** - Advanced data analysis methods
    
    **Key Features**:
    - **Mandatory filtering**: Only includes papers with BOTH EIS and AI terms
    - **Comprehensive coverage**: Searches across multiple arXiv categories
    - **Intelligent scoring**: Uses SciBERT + regex patterns for relevance scoring
    - **Full-text extraction**: Downloads and indexes complete paper content
    - **Section extraction**: Automatically identifies EIS and AI-related sections
    
    **Applications covered**:
    - Battery health monitoring and prediction
    - Corrosion detection and monitoring
    - Biomedical sensing and diagnostics
    - Fuel cell and supercapacitor analysis
    - Automated equivalent circuit fitting
    - EIS data augmentation and synthesis
    
    **AI/ML Techniques covered**:
    - Machine Learning (ML) and Deep Learning (DL)
    - Natural Language Processing (NLP) for literature analysis
    - Neural Networks (CNN, RNN, LSTM, Transformers)
    - Data-driven modeling and prediction
    - Automated feature extraction
    """)
