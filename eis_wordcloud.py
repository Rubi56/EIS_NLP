# streamlit_app.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from numba import jit, njit, prange
import logging
from pathlib import Path
import io
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="EIS & AI/ML Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 1.5rem;
}
.metric-card {
    background-color: #F8FAFC;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #3B82F6;
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
.highlight-box {
    background-color: #EBF8FF;
    border-left: 4px solid #3182CE;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.5rem;
}
.file-upload-area {
    border: 2px dashed #cbd5e1;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    background-color: #f8fafc;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def add_caption(text: str):
    """Add a styled caption below a figure"""
    st.markdown(f'<div class="figure-caption">{text}</div>', unsafe_allow_html=True)

# Numba-accelerated text processing functions
@jit(nopython=True, parallel=True)
def process_text_numba(text_array):
    """Numba-accelerated text processing for term extraction"""
    results = []
    for i in prange(len(text_array)):
        text = text_array[i]
        # Simple text cleaning
        cleaned = ""
        for char in text:
            if char.isalnum() or char.isspace():
                cleaned += char.lower()
        results.append(cleaned)
    return results

def get_db_paths():
    """
    Get database paths with multiple fallback options
    """
    paths = {}
    
    # Option 1: Check for knowledge_database folder (GitHub repository structure)
    script_dir = Path(__file__).parent
    knowledge_db_dir = script_dir / "knowledge_database"
    
    if knowledge_db_dir.exists():
        paths["Local Metadata DB"] = knowledge_db_dir / "metadata.db"
        paths["Local Universe DB"] = knowledge_db_dir / "universe.db"
        paths["Local CSV"] = knowledge_db_dir / "eis_ai_results.csv"
    
    # Option 2: Check temporary directory (from search app)
    temp_dir = Path(tempfile.gettempdir())
    paths["Temp Metadata DB"] = temp_dir / "eis_ai_metadata.db"
    paths["Temp Universe DB"] = temp_dir / "eis_ai_universe.db"
    
    # Option 3: Check for uploaded files in session state
    if 'uploaded_files' in st.session_state:
        for file_name, file_data in st.session_state.uploaded_files.items():
            if file_name.endswith('.db'):
                paths[f"Uploaded {file_name}"] = file_data
            elif file_name.endswith('.csv'):
                paths[f"Uploaded {file_name}"] = file_data
    
    return paths

def load_csv_data(csv_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} papers from CSV")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None

def download_file_from_github(url, local_path):
    """Download file from GitHub URL"""
    try:
        # Convert GitHub URL to raw download URL
        if 'github.com' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded file to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

class DatabaseManager:
    """Enhanced database manager with multiple data source support"""
    
    def __init__(self, data_source):
        self.data_source = data_source
        self.conn = None
        self.data_type = self._detect_data_type()
        logger.info(f"Data manager initialized for {data_source} (type: {self.data_type})")
    
    def _detect_data_type(self):
        """Detect if data source is CSV or DB"""
        if isinstance(self.data_source, pd.DataFrame):
            return "dataframe"
        elif str(self.data_source).endswith('.csv'):
            return "csv"
        elif str(self.data_source).endswith('.db'):
            return "database"
        else:
            return "unknown"
    
    def connect(self) -> bool:
        """Establish connection or load data"""
        try:
            if self.data_type == "dataframe":
                # Data already loaded
                return True
            elif self.data_type == "csv":
                # Load CSV into DataFrame
                self.df = pd.read_csv(self.data_source)
                logger.info(f"Loaded CSV with {len(self.df)} records")
                return True
            elif self.data_type == "database":
                # Connect to SQLite database
                if not os.path.exists(self.data_source):
                    logger.warning(f"Database file not found: {self.data_source}")
                    return False
                
                self.conn = sqlite3.connect(self.data_source)
                logger.info(f"Connected to database: {self.data_source}")
                return True
            else:
                logger.error(f"Unknown data type: {self.data_type}")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def get_papers_data(self) -> pd.DataFrame:
        """Get papers data from various sources"""
        try:
            if self.data_type == "dataframe":
                return self.df
            elif self.data_type == "csv":
                return self.df
            elif self.data_type == "database":
                return self._get_papers_from_db()
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting papers data: {e}")
            return pd.DataFrame()
    
    def _get_papers_from_db(self) -> pd.DataFrame:
        """Get papers from SQLite database"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                logger.warning("No tables found in database")
                return pd.DataFrame()
            
            # Try to find papers table
            target_table = None
            for table in tables:
                if 'paper' in table.lower():
                    target_table = table
                    break
            
            if not target_table:
                target_table = tables[0]  # Use first table if no papers table found
            
            # Get table structure
            cursor.execute(f"PRAGMA table_info({target_table})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Build query with available columns
            query = f"SELECT * FROM {target_table}"
            df = pd.read_sql_query(query, self.conn)
            
            logger.info(f"Loaded {len(df)} papers from {target_table}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching from database: {e}")
            return pd.DataFrame()
    
    def disconnect(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

class TextAnalyzer:
    """Enhanced text analysis with EIS/AI-specific processing"""
    
    def __init__(self):
        # Expanded stopwords with EIS/AI-specific terms
        self.stopwords = set([
            'using', 'used', 'use', 'paper', 'study', 'research', 'result', 'results', 'method', 'figure', 
            'table', 'shown', 'show', 'fig', 'based', 'high', 'low', 'respectively', 'obtained', 'fabricated',
            'reported', 'demonstrated', 'exhibited', 'investigated', 'characterized', 'measured', 'synthesized',
            'prepared', 'the', 'and', 'in', 'of', 'to', 'a', 'is', 'are', 'with', 'for', 'on', 'by', 'be', 'this',
            'impedance', 'spectroscopy', 'electrochemical', 'machine', 'learning', 'neural', 'network',
            'data', 'model', 'analysis', 'approach', 'system', 'process', 'time', 'can', 'also', 'however',
            'therefore', 'thus', 'may', 'could', 'would', 'should', 'might', 'one', 'two', 'first', 'second',
            'et', 'al', 'doi', 'arxiv', 'preprint', 'www', 'http', 'https', 'org', 'com', 'pdf', 'figure',
            'eis', 'ai', 'ml', 'dl'
        ])
        
        # EIS/AI-specific terms to highlight
        self.highlight_terms = set([
            'nyquist', 'bode', 'circuit', 'equivalent', 'battery', 'fuel cell', 'supercapacitor',
            'corrosion', 'electrolyte', 'electrode', 'state of charge', 'state of health', 'pinns',
            'physics-informed', 'deep learning', 'convolutional', 'recurrent', 'transformer',
            'gaussian process', 'bayesian', 'optimization', 'feature extraction', 'prediction',
            'forecasting', 'classification', 'regression', 'clustering', 'dimensionality reduction'
        ])
    
    def extract_terms(self, texts: list) -> dict:
        """Extract terms from text with Numba acceleration"""
        if not texts:
            return {}
        
        # Convert to numpy array for Numba
        text_array = np.array(texts, dtype='object')
        
        # Use Numba for text processing
        processed_texts = process_text_numba(text_array)
        
        # Count terms
        all_terms = []
        for text in processed_texts:
            words = text.split()
            # Filter short words and stopwords
            filtered_words = [word for word in words if len(word) > 2 and word not in self.stopwords]
            all_terms.extend(filtered_words)
        
        # Count frequencies
        term_counts = Counter(all_terms)
        return term_counts
    
    def extract_keyphrases(self, texts: list) -> dict:
        """Extract keyphrases (bigrams and trigrams) with EIS/AI focus"""
        if not texts:
            return {}
        
        keyphrases = []
        
        for text in texts:
            if not text or len(str(text)) < 50:
                continue
            
            # Clean text
            text_clean = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', str(text).lower())
            words = [word for word in text_clean.split() if len(word) > 2 and word not in self.stopwords]
            
            # Extract bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                # Only keep bigrams with at least one highlight term
                if any(term in bigram for term in self.highlight_terms):
                    keyphrases.append(bigram)
            
            # Extract trigrams
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                # Only keep trigrams with at least one highlight term
                if any(term in trigram for term in self.highlight_terms):
                    keyphrases.append(trigram)
        
        # Count frequencies
        phrase_counts = Counter(keyphrases)
        return phrase_counts

class VisualizationEngine:
    """Enhanced publication-quality visualizations"""
    
    def __init__(self):
        self.colors = {
            'materials': ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'],
            'properties': ['#6366F1', '#14B8A6', '#F97316', '#DC2626', '#A855F7']
        }
    
    def create_wordcloud(self, term_counts: dict, title: str = "Term Frequency Word Cloud", 
                        max_words: int = 100, colormap: str = 'viridis'):
        """Create publication-quality word cloud"""
        fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
        
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            max_words=max_words,
            colormap=colormap,
            relative_scaling=0.5,
            collocations=False
        ).generate_from_frequencies(term_counts)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=24, fontweight='bold', pad=20, fontfamily='serif')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🔬 EIS & AI/ML Analysis<br><small>Flexible Data Source Support</small></h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'extractor' not in st.session_state:
        st.session_state.extractor = TextAnalyzer()
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = VisualizationEngine()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Data Source Configuration")
        
        # Data source selection
        data_source_option = st.selectbox(
            "Select Data Source",
            ["Auto-detect", "Upload Files", "GitHub URL", "Local Files"]
        )
        
        selected_source = None
        source_type = None
        
        if data_source_option == "Auto-detect":
            st.subheader("🔍 Auto-detecting Data Sources...")
            db_paths = get_db_paths()
            
            available_sources = []
            for name, path in db_paths.items():
                if isinstance(path, Path) and path.exists():
                    available_sources.append((name, path))
                elif isinstance(path, pd.DataFrame):
                    available_sources.append((name, path))
            
            if available_sources:
                st.success(f"Found {len(available_sources)} data source(s)!")
                source_names = [name for name, _ in available_sources]
                selected_name = st.selectbox("Select source", source_names)
                selected_source = next(path for name, path in available_sources if name == selected_name)
                source_type = "auto-detected"
            else:
                st.warning("No data sources found. Please upload files or provide a GitHub URL.")
        
        elif data_source_option == "Upload Files":
            st.subheader("📁 Upload Data Files")
            
            uploaded_file = st.file_uploader(
                "Upload CSV or DB file",
                type=['csv', 'db'],
                help="Upload your eis_ai_results.csv or metadata.db file"
            )
            
            if uploaded_file:
                # Save uploaded file to session state
                file_ext = uploaded_file.name.split('.')[-1]
                temp_path = Path(tempfile.gettempdir()) / f"uploaded_{uploaded_file.name}"
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.uploaded_files[uploaded_file.name] = temp_path
                st.success(f"Uploaded {uploaded_file.name}")
                selected_source = temp_path
                source_type = "uploaded"
        
        elif data_source_option == "GitHub URL":
            st.subheader("🔗 GitHub URL")
            github_url = st.text_input(
                "Enter GitHub URL",
                placeholder="https://github.com/username/repo/blob/main/knowledge_database/eis_ai_results.csv"
            )
            
            if github_url and st.button("Download from GitHub"):
                # Determine file type and local path
                if github_url.endswith('.csv'):
                    local_path = Path(tempfile.gettempdir()) / "downloaded_results.csv"
                elif github_url.endswith('.db'):
                    local_path = Path(tempfile.gettempdir()) / "downloaded_metadata.db"
                else:
                    st.error("URL must end with .csv or .db")
                    return
                
                if download_file_from_github(github_url, local_path):
                    st.success(f"Downloaded to {local_path}")
                    selected_source = local_path
                    source_type = "downloaded"
                else:
                    st.error("Failed to download file")
        
        elif data_source_option == "Local Files":
            st.subheader("📂 Local File Paths")
            local_path = st.text_input(
                "Enter full path to file",
                placeholder="/path/to/your/knowledge_database/eis_ai_results.csv"
            )
            
            if local_path and Path(local_path).exists():
                selected_source = Path(local_path)
                source_type = "local"
                st.success(f"Found file at {local_path}")
            elif local_path:
                st.error("File not found at specified path")
        
        # Analysis parameters
        if selected_source:
            st.subheader("⚙️ Analysis Parameters")
            max_papers = st.slider("Max papers to analyze", 10, 500, 200, 10)
            min_term_freq = st.slider("Min term frequency", 1, 20, 3)
            
            # Word cloud settings
            st.subheader("☁️ Word Cloud Settings")
            max_words = st.slider("Max words in cloud", 10, 300, 150)
            colormap = st.selectbox("Color scheme", 
                                   ["viridis", "plasma", "inferno", "magma", "Blues", "Greens", "Reds"])
            
            # Processing options
            st.subheader("🔧 Processing Options")
            use_numba = st.checkbox("Enable Numba JIT Acceleration", value=True)
            extract_keyphrases = st.checkbox("Extract Keyphrases (Bigrams/Trigrams)", value=True)
            
            # Action button
            analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    # Main analysis workflow
    if 'selected_source' in locals() and selected_source and analyze_btn:
        with st.spinner(f"🔬 Analyzing EIS & AI/ML data..."):
            try:
                # Initialize data manager
                data_manager = DatabaseManager(selected_source)
                
                if not data_manager.connect():
                    st.error("Failed to connect/load data source!")
                    return
                
                # Load papers
                st.text("📥 Loading papers...")
                papers_df = data_manager.get_papers_data()
                
                if papers_df.empty:
                    st.error("No papers found in data source!")
                    data_manager.disconnect()
                    return
                
                # Limit papers for performance
                papers_df = papers_df.head(max_papers).copy()
                
                # Extract text content
                st.text("📝 Extracting text content...")
                texts = []
                
                for idx, row in papers_df.iterrows():
                    # Try different text columns
                    text_content = None
                    
                    for col in ['full_text', 'abstract', 'content', 'summary']:
                        if col in papers_df.columns and pd.notna(row[col]):
                            text_content = str(row[col])
                            break
                    
                    if text_content and len(text_content) > 50:
                        texts.append(text_content)
                
                if not texts:
                    st.error("No valid text content found!")
                    data_manager.disconnect()
                    return
                
                # Term extraction
                st.text("⚡ Extracting terms...")
                if extract_keyphrases:
                    term_counts = st.session_state.extractor.extract_keyphrases(texts)
                else:
                    term_counts = st.session_state.extractor.extract_terms(texts)
                
                # Filter by frequency
                filtered_terms = {term: count for term, count in term_counts.items() 
                                if count >= min_term_freq and len(term.split()) <= 3}
                
                # Store results
                st.session_state.analysis_results = {
                    'papers': papers_df,
                    'term_counts': filtered_terms,
                    'source_type': source_type,
                    'source_path': str(selected_source)
                }
                
                st.success(f"✅ Analysis complete! Found {len(filtered_terms)} unique terms in {len(texts)} papers.")
                
                # Show summary metrics
                with st.expander("📊 Analysis Summary"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Papers Analyzed", len(texts))
                    col2.metric("Unique Terms", len(filtered_terms))
                    col3.metric("Data Source", source_type)
                
                data_manager.disconnect()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error(f"Analysis failed: {str(e)}", exc_info=True)
    
    # Results display
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        term_counts = results['term_counts']
        papers_df = results['papers']
        
        st.markdown("### 📊 Analysis Results")
        st.info(f"Data source: {results['source_type']} - {results['source_path']}")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Word Cloud", "Term List", "Data Preview"])
        
        with tab1:
            if term_counts:
                st.subheader("☁️ Term Frequency Word Cloud")
                
                # Create word cloud
                fig = st.session_state.viz_engine.create_wordcloud(
                    term_counts,
                    f"Key Terms in EIS & AI/ML Literature ({len(papers_df)} Papers)",
                    max_words=max_words,
                    colormap=colormap
                )
                
                st.pyplot(fig)
                
                # Download option
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "📥 Download Word Cloud",
                    buf.getvalue(),
                    "wordcloud_eis_ai.png",
                    "image/png"
                )
            else:
                st.info("No terms available for word cloud. Try reducing the minimum frequency threshold.")
        
        with tab2:
            st.subheader("📋 Term Frequency List")
            
            if term_counts:
                # Convert to DataFrame
                terms_df = pd.DataFrame({
                    'Term': list(term_counts.keys()),
                    'Frequency': list(term_counts.values())
                }).sort_values('Frequency', ascending=False)
                
                # Display with filtering
                min_freq_filter = st.slider("Filter by minimum frequency", 1, max(terms_df['Frequency']), 1)
                filtered_df = terms_df[terms_df['Frequency'] >= min_freq_filter]
                
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download options
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "terms.csv", "text/csv")
            else:
                st.info("No terms available.")
        
        with tab3:
            st.subheader("📄 Data Preview")
            st.write(f"Shape: {papers_df.shape}")
            st.dataframe(papers_df.head(10), use_container_width=True)
            
            # Column info
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Column': papers_df.columns,
                'Data Type': papers_df.dtypes.values,
                'Non-Null Count': papers_df.count().values,
                'Sample Values': [str(papers_df[col].iloc[0]) if len(papers_df) > 0 and pd.notna(papers_df[col].iloc[0]) else 'N/A' for col in papers_df.columns]
            })
            st.dataframe(col_info, use_container_width=True)

if __name__ == "__main__":
    main()
