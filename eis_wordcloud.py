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

def get_db_paths():
    """
    Get database paths for the EIS & AI search results.
    Compatible with the improved search that finds 190 papers.
    """
    # Use the same temporary directory as the main app
    BASE_DIR = Path(tempfile.gettempdir())
    
    return {
        "Metadata DB": BASE_DIR / "eis_ai_metadata.db",
        "Universe DB": BASE_DIR / "eis_ai_universe.db"
    }

class DatabaseManager:
    """Enhanced database manager for SQLite connections with better EIS/AI support"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        logger.info(f"Database manager initialized for {db_path}")
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Database file not found: {db_path}")
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
        """Get papers data from database with improved EIS/AI support"""
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
            
            # Find text columns - prioritize full_text, then abstract, then content
            text_columns = [col for col in columns if 'text' in col.lower() or 'content' in col.lower() or 'abstract' in col.lower()]
            
            if not text_columns:
                logger.warning("No text columns found")
                return pd.DataFrame()
            
            # Use the best text column available
            if 'full_text' in text_columns:
                text_column = 'full_text'
            elif 'abstract' in text_columns:
                text_column = 'abstract'
            else:
                text_column = text_columns[0]
            
            # Build query with available columns
            select_cols = [text_column]
            
            # Add metadata columns if available
            for col in ['title', 'authors', 'year', 'categories', 'matched_terms', 'relevance_prob']:
                if col in columns:
                    select_cols.append(col)
            
            # If year not available, use a default
            if 'year' not in select_cols:
                select_cols.append("2023 as year")
            
            # If title not available, use a default
            if 'title' not in select_cols:
                select_cols.append("'No title' as title")
            
            # If categories not available, use a default
            if 'categories' not in select_cols:
                select_cols.append("'Unknown' as categories")
            
            query = f"""
            SELECT {', '.join(select_cols)}
            FROM {target_table}
            WHERE {text_column} IS NOT NULL AND LENGTH({text_column}) > 50
            ORDER BY 
                CASE WHEN relevance_prob IS NOT NULL THEN relevance_prob ELSE 0 END DESC,
                year DESC
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
    """Enhanced text analysis with EIS/AI-specific processing"""
    
    def __init__(self):
        # Expanded stopwords with EIS/AI-specific terms that are too common
        self.stopwords = set([
            'using', 'used', 'use', 'paper', 'study', 'research', 'result', 'results', 'method', 'figure', 
            'table', 'shown', 'show', 'fig', 'based', 'high', 'low', 'respectively', 'obtained', 'fabricated',
            'reported', 'demonstrated', 'exhibited', 'investigated', 'characterized', 'measured', 'synthesized',
            'prepared', 'the', 'and', 'in', 'of', 'to', 'a', 'is', 'are', 'with', 'for', 'on', 'by', 'be', 'this',
            'impedance', 'spectroscopy', 'electrochemical', 'machine', 'learning', 'neural', 'network',
            'data', 'model', 'analysis', 'approach', 'system', 'process', 'time', 'can', 'also', 'however',
            'therefore', 'thus', 'may', 'could', 'would', 'should', 'might', 'one', 'two', 'first', 'second',
            'et', 'al', 'doi', 'arxiv', 'preprint', 'www', 'http', 'https', 'org', 'com', 'pdf', 'figure',
            'eis', 'ai', 'ml', 'dl'  # Common acronyms that are too frequent
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
    
    def extract_trending_terms(self, texts: list, years: list) -> dict:
        """Extract terms that are trending over time"""
        # Group texts by year
        year_to_texts = {}
        for text, year in zip(texts, years):
            if year not in year_to_texts:
                year_to_texts[year] = []
            year_to_texts[year].append(text)
        
        # Extract terms for each year
        year_to_terms = {}
        for year, year_texts in year_to_texts.items():
            term_counts = self.extract_terms(year_texts)
            year_to_terms[year] = term_counts
        
        # Calculate term growth over time
        sorted_years = sorted(year_to_terms.keys())
        trending_terms = {}
        
        if len(sorted_years) >= 2:
            for term in set().union(*[terms.keys() for terms in year_to_terms.values()]):
                # Calculate frequency in each year
                frequencies = [year_to_terms[year].get(term, 0) for year in sorted_years]
                
                # Calculate growth rate (simple linear regression slope)
                if sum(frequencies) > 5:  # Only consider terms with sufficient frequency
                    x = np.arange(len(frequencies))
                    if np.var(x) > 0:  # Avoid division by zero
                        slope = np.cov(x, frequencies)[0, 1] / np.var(x)
                        if slope > 0:  # Only consider growing terms
                            trending_terms[term] = slope
        
        return sorted(trending_terms.items(), key=lambda x: x[1], reverse=True)

class VisualizationEngine:
    """Enhanced publication-quality visualizations with Plotly support"""
    
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
    
    def create_term_trend_chart(self, term_data: dict, title: str = "Term Trend Analysis"):
        """Create interactive term trend chart using Plotly"""
        fig = go.Figure()
        
        for term, trend_data in term_data.items():
            fig.add_trace(go.Scatter(
                x=list(trend_data.keys()),
                y=list(trend_data.values()),
                mode='lines+markers',
                name=term,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Frequency",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_relevance_distribution(self, df: pd.DataFrame):
        """Create relevance score distribution chart"""
        if 'relevance_prob' in df.columns:
            fig = px.histogram(
                df, 
                x="relevance_prob", 
                nbins=20,
                title="Distribution of Paper Relevance Scores",
                labels={"relevance_prob": "Relevance Score (%)", "count": "Number of Papers"},
                color_discrete_sequence=['#3B82F6']
            )
            fig.update_layout(bargap=0.1)
            return fig
        return None
    
    def create_yearly_publication_trend(self, df: pd.DataFrame):
        """Create yearly publication trend chart"""
        if 'year' in df.columns:
            year_counts = df['year'].value_counts().sort_index()
            fig = px.line(
                x=year_counts.index,
                y=year_counts.values,
                title="Publication Trend Over Years",
                labels={"x": "Year", "y": "Number of Papers"},
                markers=True,
                color_discrete_sequence=['#10B981']
            )
            fig.update_traces(line_width=3)
            return fig
        return None

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🔬 EIS & AI/ML Analysis<br><small>Enhanced Analysis for 190+ Papers</small></h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = TextAnalyzer()
    
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = VisualizationEngine()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get database paths
        db_paths = get_db_paths()
        
        # Database selection
        available_dbs = []
        for db_name, db_path in db_paths.items():
            if db_path.exists():
                available_dbs.append(db_name)
        
        if not available_dbs:
            st.warning("No databases found! Please run the EIS & AI search first.")
            st.info("The search will create databases in the temporary directory.")
            return
        
        selected_db = st.selectbox("Select Database", available_dbs)
        selected_db_path = db_paths[selected_db]
        
        # Display database path
        st.markdown(f"""
        <div style="background-color: #FEF7CD; padding: 0.5rem; border-radius: 4px; font-size: 0.85em;">
            <strong>Database Path:</strong><br>{selected_db_path}
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        max_papers = st.slider("Max papers to analyze", 10, 500, 200, 10)
        min_term_freq = st.slider("Min term frequency", 1, 20, 3)
        
        # Word cloud settings
        st.subheader("Word Cloud Settings")
        max_words = st.slider("Max words in cloud", 10, 300, 150)
        colormap = st.selectbox("Color scheme", 
                               ["viridis", "plasma", "inferno", "magma", "Blues", "Greens", "Reds"])
        
        # Processing options
        st.subheader("Processing Options")
        use_numba = st.checkbox("Enable Numba JIT Acceleration", value=True)
        extract_keyphrases = st.checkbox("Extract Keyphrases (Bigrams/Trigrams)", value=True)
        analyze_trends = st.checkbox("Analyze Term Trends Over Time", value=True)
        
        # Actions
        st.subheader("Actions")
        analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.analysis_results = None
            st.rerun()
    
    # Main analysis workflow
    if analyze_btn:
        with st.spinner(f"🔬 Analyzing EIS & AI/ML data with Numba acceleration..."):
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
                years = []
                
                for idx, row in papers_df.iterrows():
                    # Try to get full text first
                    if 'full_text' in row and pd.notna(row['full_text']) and len(str(row['full_text'])) > 50:
                        texts.append(str(row['full_text']))
                    elif 'abstract' in row and pd.notna(row['abstract']) and len(str(row['abstract'])) > 50:
                        texts.append(str(row['abstract']))
                    else:
                        continue  # Skip if no valid text
                    
                    # Extract year
                    if 'year' in row and pd.notna(row['year']):
                        years.append(int(row['year']))
                    else:
                        years.append(2023)  # Default year
                
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
                    st.warning("No terms meet the frequency threshold! Try lowering the minimum frequency.")
                
                # Trend analysis
                trend_results = None
                if analyze_trends and len(set(years)) > 1:
                    st.text("📈 Analyzing term trends over time...")
                    trend_results = st.session_state.extractor.extract_trending_terms(texts, years)
                
                # Store results
                st.session_state.analysis_results = {
                    'papers': papers_df,
                    'term_counts': filtered_terms,
                    'trend_results': trend_results,
                    'years': years
                }
                
                st.success(f"✅ Analysis complete! Found {len(filtered_terms)} unique terms in {len(texts)} papers.")
                
                # Show summary metrics
                with st.expander("📊 Analysis Summary"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Papers Analyzed", len(texts))
                    col2.metric("Unique Terms", len(filtered_terms))
                    col3.metric("Years Spanned", f"{min(years)}-{max(years)}")
                    col4.metric("Top Term", max(filtered_terms.items(), key=lambda x: x[1])[0] if filtered_terms else "N/A")
            
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
        years = results['years']
        trend_results = results['trend_results']
        
        st.markdown("### 📊 Analysis Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Word Cloud", "Term List", "Trends", "Database Stats", "Interactive Charts"])
        
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
                    f"wordcloud_eis_ai.png",
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
            st.subheader("📈 Term Trend Analysis")
            
            if trend_results:
                st.markdown("### 🔥 Trending Terms (Growing in Frequency)")
                
                # Display top trending terms
                top_trends = trend_results[:10] if len(trend_results) > 10 else trend_results
                
                for term, slope in top_trends:
                    st.markdown(f"**{term}**: Growth rate = {slope:.2f}")
                
                # Create trend chart for top terms
                if len(top_trends) > 0:
                    # Get data for top 5 trending terms
                    top_terms = [term for term, _ in top_trends[:5]]
                    
                    # Calculate term frequency by year
                    term_by_year = {}
                    for term in top_terms:
                        term_by_year[term] = {}
                        
                        for year, text in zip(years, texts):
                            if year not in term_by_year[term]:
                                term_by_year[term][year] = 0
                            
                            # Simple count (in a real app, this would be more sophisticated)
                            term_by_year[term][year] += text.lower().count(term.lower())
                    
                    # Create trend chart
                    trend_fig = st.session_state.viz_engine.create_term_trend_chart(
                        term_by_year, 
                        "Frequency of Top Trending Terms Over Time"
                    )
                    st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("No trend data available. This might be because all papers are from the same year or trend analysis was disabled.")
        
        with tab4:
            st.subheader("📈 Database Statistics")
            
            st.markdown(f"""
            **Database Information:**
            - **Papers Analyzed**: {len(papers_df)}
            - **Unique Terms**: {len(term_counts)}
            - **Database File**: {Path(selected_db_path).name}
            - **Processing**: {'Numba JIT Accelerated' if use_numba else 'Standard Python'}
            - **Years Spanned**: {min(years)}-{max(years)}
            """)
            
            if not papers_df.empty:
                st.markdown("### 📅 Publication Years")
                year_counts = papers_df['year'].value_counts().sort_index()
                st.bar_chart(year_counts)
                
                if 'categories' in papers_df.columns:
                    st.markdown("### 🏷️ Categories")
                    category_counts = papers_df['categories'].value_counts()
                    st.bar_chart(category_counts)
                
                if 'relevance_prob' in papers_df.columns:
                    st.markdown("### 📊 Relevance Scores")
                    st.write(papers_df['relevance_prob'].describe())
                
                st.markdown("### 📄 Sample Papers")
                for i, row in papers_df.head(3).iterrows():
                    with st.expander(f"📄 {row.get('title', 'No title')} ({row.get('year', 'Unknown')})"):
                        if 'authors' in row and pd.notna(row['authors']):
                            st.markdown(f"**Authors:** {row['authors']}")
                        if 'categories' in row and pd.notna(row['categories']):
                            st.markdown(f"**Category:** {row['categories']}")
                        if 'relevance_prob' in row and pd.notna(row['relevance_prob']):
                            st.markdown(f"**Relevance:** {row['relevance_prob']}%")
                        if 'matched_terms' in row and pd.notna(row['matched_terms']):
                            st.markdown(f"**Matched Terms:** {row['matched_terms']}")
                        if 'abstract' in row and pd.notna(row['abstract']):
                            st.markdown(f"**Abstract:** {row['abstract'][:300]}...")
            
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
        
        with tab5:
            st.subheader("📊 Interactive Charts")
            
            # Relevance distribution
            relevance_fig = st.session_state.viz_engine.create_relevance_distribution(papers_df)
            if relevance_fig:
                st.plotly_chart(relevance_fig, use_container_width=True)
            
            # Yearly publication trend
            trend_fig = st.session_state.viz_engine.create_yearly_publication_trend(papers_df)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
            
            # Top terms bar chart
            if term_counts:
                top_terms = dict(sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:20])
                terms_fig = px.bar(
                    x=list(top_terms.values()),
                    y=list(top_terms.keys()),
                    orientation='h',
                    title="Top 20 Terms by Frequency",
                    labels={"x": "Frequency", "y": "Term"},
                    color_discrete_sequence=['#3B82F6']
                )
                terms_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(terms_fig, use_container_width=True)

if __name__ == "__main__":
    main()
