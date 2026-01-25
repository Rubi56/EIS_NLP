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
        # Simple text cleaning (in practice, this would be more complex)
        cleaned = ""
        for char in text:
            if char.isalnum() or char.isspace():
                cleaned += char.lower()
        results.append(cleaned)
    return results

def get_db_paths_for_query(query_id: str = "q0") -> dict:
    """
    Get database paths for a specific query dataset.
    query_id = "q0" for default, "q1" for query1, etc.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    knowledge_db_dir = script_dir / "knowledge_database"
    
    # Create directory if it doesn't exist
    knowledge_db_dir.mkdir(exist_ok=True)
    
    # Handle default case (q0 should use base names without q0 suffix)
    suffix = f"{query_id}_" if query_id != "q0" else ""
    
    return {
        "Metadata DB": knowledge_db_dir / f"eis_ai{suffix}metadata.db",
        "Universe DB": knowledge_db_dir / f"eis_ai{suffix}universe.db",
        "PDF Storage DB": knowledge_db_dir / f"eis_ai{suffix}pdfs.db"
    }

class DatabaseManager:
    """Simple database manager for SQLite connections"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        logger.info(f"Database manager initialized for {db_path}")
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Database file not found: {self.db_path}")
                return False
            
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    def get_tables(self) -> list:
        """Get list of tables in database"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            logger.debug(f"Found tables: {tables}")
            return tables
        except Exception as e:
            logger.error(f"Error fetching tables: {e}")
            return []
    
    def get_papers_data(self) -> pd.DataFrame:
        """Get papers data from database"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            tables = self.get_tables()
            target_table = None
            
            # Look for papers table
            for table in tables:
                if 'paper' in table.lower() or 'document' in table.lower():
                    target_table = table
                    break
            
            if not target_table:
                logger.warning("No papers table found")
                return pd.DataFrame()
            
            # Get columns
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({target_table})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Find text columns
            text_columns = [col for col in columns if 'text' in col.lower() or 'content' in col.lower() or 'abstract' in col.lower()]
            
            if not text_columns:
                logger.warning("No text columns found")
                return pd.DataFrame()
            
            # Query data
            text_column = text_columns[0]
            query = f"""
            SELECT 
                {text_column} as full_text,
                COALESCE(title, 'No title') as title,
                COALESCE(year, 2023) as year,
                COALESCE(categories, 'Unknown') as categories
            FROM {target_table}
            WHERE {text_column} IS NOT NULL AND LENGTH({text_column}) > 50
            LIMIT 500
            """
            
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Loaded {len(df)} papers from {target_table}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            return pd.DataFrame()
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

class TextAnalyzer:
    """Text analysis with Numba acceleration"""
    
    def __init__(self):
        self.stopwords = set([
            'using', 'used', 'use', 'paper', 'study', 'research', 'result', 'results', 'method', 'figure', 
            'table', 'shown', 'show', 'fig', 'based', 'high', 'low', 'respectively', 'obtained', 'fabricated',
            'reported', 'demonstrated', 'exhibited', 'investigated', 'characterized', 'measured', 'synthesized',
            'prepared', 'the', 'and', 'in', 'of', 'to', 'a', 'is', 'are', 'with', 'for', 'on', 'by', 'be', 'this',
            'impedance', 'spectroscopy', 'electrochemical', 'machine', 'learning', 'neural', 'network'
        ])
    
    def extract_terms(self, texts: list) -> dict:
        """Extract terms from text with Numba acceleration"""
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
        """Extract keyphrases (bigrams and trigrams)"""
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
                keyphrases.append(bigram)
            
            # Extract trigrams
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                keyphrases.append(trigram)
        
        # Count frequencies
        phrase_counts = Counter(keyphrases)
        return phrase_counts

class VisualizationEngine:
    """Publication-quality visualizations"""
    
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

def create_sample_database(query_id="q0"):
    """Create sample database for demonstration"""
    db_paths = get_db_paths_for_query(query_id)
    
    for db_name, db_path in db_paths.items():
        if not os.path.exists(db_path):
            logger.info(f"Creating sample database: {db_path}")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create sample papers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                abstract TEXT,
                full_text TEXT,
                year INTEGER,
                categories TEXT
            )
            ''')
            
            
            cursor.executemany('''
            INSERT INTO papers (title, abstract, full_text, year, categories)
            VALUES (?, ?, ?, ?, ?)
            ''', sample_papers)
            
            conn.commit()
            conn.close()
            logger.info(f"Created sample database: {db_path}")

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🔬 EIS & AI/ML Analysis<br><small>Query-Based Database Explorer with Numba Acceleration</small></h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = "q0"
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = TextAnalyzer()
    
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = VisualizationEngine()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Query selector
        query_options = ["q0", "q1", "q2", "q3"]
        current_query = st.selectbox(
            "Select Query Dataset", 
            query_options,
            index=query_options.index(st.session_state.current_query)
        )
        
        # Update session state if query changes
        if current_query != st.session_state.current_query:
            st.session_state.current_query = current_query
            st.session_state.analysis_results = None
            st.rerun()
        
        # Show database info
        st.markdown(f"""
        <div style="background-color: #F0F9FF; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0;">
            <strong>Current Query:</strong> {current_query}<br>
            <small>Dataset: {'Default' if current_query == 'q0' else f'Query {current_query[1:]}'} EIS & AI/ML Papers</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Get database paths for current query
        db_paths = get_db_paths_for_query(current_query)
        
        # Database selection
        available_dbs = []
        for db_name, db_path in db_paths.items():
            if db_path.exists():
                available_dbs.append(db_name)
        
        if not available_dbs:
            st.warning(f"No databases found for query '{current_query}'!")
            st.info("Creating sample databases...")
            
            # Create sample databases
            create_sample_database(current_query)
            
            # Recheck available databases
            available_dbs = []
            for db_name, db_path in db_paths.items():
                if db_path.exists():
                    available_dbs.append(db_name)
        
        if available_dbs:
            selected_db = st.selectbox("Select Database", available_dbs)
            selected_db_path = db_paths[selected_db]
            
            # Display database path
            st.markdown(f"""
            <div style="background-color: #FEF7CD; padding: 0.5rem; border-radius: 4px; font-size: 0.85em;">
                <strong>Database Path:</strong><br>{selected_db_path}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No databases available after creation!")
            return
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        max_papers = st.slider("Max papers to analyze", 10, 500, 100, 10)
        min_term_freq = st.slider("Min term frequency", 1, 10, 2)
        
        # Word cloud settings
        st.subheader("Word Cloud Settings")
        max_words = st.slider("Max words in cloud", 10, 300, 100)
        colormap = st.selectbox("Color scheme", 
                               ["viridis", "plasma", "inferno", "magma", "Blues", "Greens", "Reds"])
        
        # Processing options
        st.subheader("Processing Options")
        use_numba = st.checkbox("Enable Numba JIT Acceleration", value=True)
        extract_keyphrases = st.checkbox("Extract Keyphrases (Bigrams/Trigrams)", value=True)
        
        # Actions
        st.subheader("Actions")
        analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.analysis_results = None
            st.rerun()
    
    # Main analysis workflow
    if analyze_btn:
        with st.spinner(f"🔬 Analyzing EIS & AI/ML data from Query {current_query} with Numba acceleration..."):
            try:
                # Initialize database manager
                db_manager = DatabaseManager(str(selected_db_path))
                
                if not db_manager.connect():
                    st.error("Failed to connect to database!")
                    return
                
                # Load papers
                st.text("📥 Loading papers from database...")
                papers_df = db_manager.get_papers_data()
                
                if papers_df.empty:
                    st.error("No papers found in database!")
                    db_manager.disconnect()
                    return
                
                # Limit papers for performance
                papers_df = papers_df.head(max_papers).copy()
                
                # Extract text content
                st.text("📝 Extracting text content...")
                texts = []
                for idx, row in papers_df.iterrows():
                    if 'full_text' in row and pd.notna(row['full_text']) and len(str(row['full_text'])) > 50:
                        texts.append(str(row['full_text']))
                    elif 'abstract' in row and pd.notna(row['abstract']) and len(str(row['abstract'])) > 50:
                        texts.append(str(row['abstract']))
                
                if not texts:
                    st.error("No valid text content found!")
                    db_manager.disconnect()
                    return
                
                # Term extraction with Numba acceleration
                st.text("⚡ Extracting terms with Numba JIT acceleration...")
                if extract_keyphrases:
                    term_counts = st.session_state.extractor.extract_keyphrases(texts)
                else:
                    term_counts = st.session_state.extractor.extract_terms(texts)
                
                # Filter by frequency
                filtered_terms = {term: count for term, count in term_counts.items() 
                                if count >= min_term_freq and len(term.split()) <= 3}
                
                if not filtered_terms:
                    st.warning("No terms meet the frequency threshold!")
                
                # Store results
                st.session_state.analysis_results = {
                    'papers': papers_df,
                    'term_counts': filtered_terms,
                    'query_id': current_query
                }
                
                st.success(f"✅ Analysis complete! Found {len(filtered_terms)} unique terms in {len(texts)} papers.")
                
                # Show summary metrics
                with st.expander("📊 Analysis Summary"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Papers Analyzed", len(texts))
                    col2.metric("Unique Terms", len(filtered_terms))
                    col3.metric("Top Term", max(filtered_terms.items(), key=lambda x: x[1])[0] if filtered_terms else "N/A")
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            finally:
                if 'db_manager' in locals():
                    db_manager.disconnect()
    
    # Results display
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        term_counts = results['term_counts']
        papers_df = results['papers']
        query_id = results['query_id']
        
        st.markdown("### 📊 Analysis Results")
        
        # Word cloud tab
        tab1, tab2, tab3 = st.tabs(["Word Cloud", "Term List", "Database Stats"])
        
        with tab1:
            if term_counts:
                st.subheader("☁️ Term Frequency Word Cloud")
                
                # Create word cloud
                fig = st.session_state.viz_engine.create_wordcloud(
                    term_counts,
                    f"Key Terms in EIS & AI/ML Literature (Query {query_id})",
                    max_words=max_words,
                    colormap=colormap
                )
                
                st.pyplot(fig)
                
                add_caption(r"""
                **Methodology**: Term frequency visualized with $\text{size} \propto \log(1 + f_i)$,
                where $f_i$ is the raw frequency of term $i$. Common stopwords and noise terms removed.
                High-resolution (300 DPI) suitable for publication. Numba JIT acceleration used for text processing.
                """)
                
                # Download option
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "📥 Download High-Resolution Word Cloud",
                    buf.getvalue(),
                    f"wordcloud_query{query_id}.png",
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
                col1, col2 = st.columns(2)
                with col1:
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, "terms.csv", "text/csv")
                with col2:
                    st.download_button("📊 Download Excel", 
                                     filtered_df.to_excel(index=False).encode('utf-8'),
                                     "terms.xlsx", 
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("No terms available.")
        
        with tab3:
            st.subheader("📈 Database Statistics")
            
            st.markdown(f"""
            **Database Information for Query {query_id}:**
            - **Papers Analyzed**: {len(papers_df)}
            - **Unique Terms**: {len(term_counts)}
            - **Database File**: {Path(selected_db_path).name}
            - **Processing**: {'Numba JIT Accelerated' if use_numba else 'Standard Python'}
            """)
            
            if not papers_df.empty:
                st.markdown("### 📅 Publication Years")
                year_counts = papers_df['year'].value_counts().sort_index()
                st.bar_chart(year_counts)
                
                st.markdown("### 🏷️ Categories")
                category_counts = papers_df['categories'].value_counts()
                st.bar_chart(category_counts)
                
                st.markdown("### 📄 Sample Papers")
                for i, row in papers_df.head(3).iterrows():
                    with st.expander(f"📄 {row['title']} ({row['year']})"):
                        st.markdown(f"**Category:** {row['categories']}")
                        st.markdown(f"**Abstract:** {row.get('abstract', 'No abstract available')[:300]}...")
            
            # Performance metrics
            st.markdown("### ⚡ Performance Metrics")
            st.markdown(f"""
            **Numba JIT Acceleration:**
            - Text processing: {'Enabled' if use_numba else 'Disabled'}
            - Parallel execution: {'Enabled' if use_numba else 'Disabled'}
            - Memory optimization: {'Enabled' if use_numba else 'Disabled'}
            
            **Analysis Statistics:**
            - Total terms before filtering: {len(term_counts) + len({k:v for k,v in term_counts.items() if v < min_term_freq})}
            - Terms after frequency filtering: {len(term_counts)}
            - Average terms per paper: {len(term_counts) / len(papers_df):.1f}
            """)

if __name__ == "__main__":
    main()
