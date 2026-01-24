import arxiv
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
import io

# ==============================
# STREAMLIT CONFIGURATION
# ==============================
st.set_page_config(
    page_title="EIS & AI Research Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 EIS with AI/ML Research Explorer")
st.markdown("""
This tool searches arXiv for papers combining **Electrochemical Impedance Spectroscopy (EIS)** with **Artificial Intelligence/Machine Learning**.
- **EIS**: Nyquist plots, equivalent circuits, impedance analysis
- **AI/ML**: Machine learning, neural networks, data-driven models
- **Applications**: Batteries, corrosion, sensors, biomedical
""")

# ==============================
# SESSION STATE
# ==============================
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "downloaded_pdfs" not in st.session_state:
    st.session_state.downloaded_pdfs = {}
if "papers_df" not in st.session_state:
    st.session_state.papers_df = None

def update_log(message):
    """Add a timestamped message to the log buffer."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)

# ==============================
# SIMPLE SCORING FUNCTION
# ==============================
def score_paper(title, abstract):
    """Simple scoring based on term matching."""
    text = f"{title} {abstract}".lower()
    
    # EIS terms
    eis_terms = [
        'electrochemical impedance', 'eis', 'impedance spectroscopy',
        'nyquist', 'bode', 'equivalent circuit', 'charge transfer',
        'constant phase element', 'cpe', 'warburg'
    ]
    
    # AI terms
    ai_terms = [
        'machine learning', 'deep learning', 'neural network',
        'artificial intelligence', 'ai', 'ml', 'data-driven',
        'neural', 'learning', 'regression', 'classification',
        'random forest', 'svm', 'gradient boosting'
    ]
    
    # Count matches
    eis_score = sum(1 for term in eis_terms if term in text)
    ai_score = sum(1 for term in ai_terms if term in text)
    
    # Calculate relevance (0-100)
    if eis_score > 0 and ai_score > 0:
        base_score = 50 + (eis_score * 10) + (ai_score * 10)
    elif eis_score > 0 or ai_score > 0:
        base_score = 30 + (max(eis_score, ai_score) * 10)
    else:
        base_score = 0
    
    # Cap at 100
    return min(base_score, 100), eis_score, ai_score

# ==============================
# DATABASE FUNCTIONS
# ==============================
def init_metadata_db():
    """Initialize SQLite database."""
    db_file = Path(tempfile.gettempdir()) / "eis_ai_metadata.db"
    try:
        conn = sqlite3.connect(db_file)
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
                relevance INTEGER,
                eis_score INTEGER,
                ai_score INTEGER
            )
        """)
        conn.commit()
        conn.close()
        return db_file
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

# ==============================
# ARXIV QUERY FUNCTION
# ==============================
def query_arxiv(query, categories, max_results, start_year, end_year):
    """Query arXiv API."""
    try:
        update_log(f"Searching arXiv with query: {query[:100]}...")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # Get extra for filtering
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        seen_ids = set()
        
        for result in client.results(search):
            # Year filter
            if not (start_year <= result.published.year <= end_year):
                continue
            
            # Avoid duplicates
            paper_id = result.get_short_id()
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            
            # Score paper
            relevance, eis_score, ai_score = score_paper(result.title, result.summary)
            
            # Only include papers with some relevance
            if relevance < 20:  # Lower threshold to get more papers
                continue
            
            papers.append({
                "id": paper_id,
                "title": result.title,
                "authors": ", ".join([author.name for author in result.authors]),
                "year": result.published.year,
                "categories": ", ".join(result.categories),
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "relevance": relevance,
                "eis_score": eis_score,
                "ai_score": ai_score,
                "has_eis": eis_score > 0,
                "has_ai": ai_score > 0
            })
            
            if len(papers) >= max_results:
                break
        
        # Sort by relevance
        papers.sort(key=lambda x: x["relevance"], reverse=True)
        update_log(f"Found {len(papers)} relevant papers")
        return papers
        
    except Exception as e:
        update_log(f"Query failed: {str(e)}")
        st.error(f"Error: {str(e)}")
        return []

# ==============================
# PDF DOWNLOAD
# ==============================
def download_pdf(pdf_url, paper_id):
    """Download PDF file."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        st.session_state.downloaded_pdfs[paper_id] = response.content
        update_log(f"Downloaded PDF: {paper_id}")
        return True
    except Exception as e:
        update_log(f"Download failed: {str(e)}")
        return False

# ==============================
# SIDEBAR CONTROLS
# ==============================
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    # Query options
    query_option = st.selectbox(
        "Search Type",
        ["EIS AND AI/ML", "EIS only", "AI/ML only", "Custom"]
    )
    
    if query_option == "EIS AND AI/ML":
        query = '(electrochemical impedance OR EIS OR impedance spectroscopy) AND (machine learning OR artificial intelligence OR deep learning OR neural network)'
    elif query_option == "EIS only":
        query = 'electrochemical impedance OR EIS OR impedance spectroscopy'
    elif query_option == "AI/ML only":
        query = 'machine learning OR artificial intelligence OR deep learning OR neural network'
    else:
        query = st.text_area(
            "Custom Query",
            value='(electrochemical impedance OR EIS) AND (machine learning OR AI)',
            height=80
        )
    
    # Categories
    categories = st.multiselect(
        "arXiv Categories",
        ["physics.chem-ph", "cond-mat.mtrl-sci", "cs.LG", "cs.AI", "physics.app-ph"],
        default=["physics.chem-ph", "cond-mat.mtrl-sci", "cs.LG"]
    )
    
    # Limits
    col1, col2 = st.columns(2)
    with col1:
        max_results = st.slider("Max Results", 10, 100, 50)
    with col2:
        current_year = datetime.now().year
        start_year = st.number_input("Start Year", 2000, current_year, 2010)
        end_year = st.number_input("End Year", start_year, current_year, current_year)
    
    # Filters
    st.markdown("---")
    st.subheader("Filters")
    min_relevance = st.slider("Min Relevance", 0, 100, 20)
    
    col1, col2 = st.columns(2)
    with col1:
        require_eis = st.checkbox("Require EIS", value=True)
    with col2:
        require_ai = st.checkbox("Require AI", value=True)
    
    # Search button
    search_button = st.button("🔍 Search arXiv", type="primary", use_container_width=True)

# ==============================
# MAIN CONTENT
# ==============================
if search_button:
    if not categories:
        st.error("Please select at least one category")
    else:
        with st.spinner("Searching arXiv..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("No papers found. Try broadening your search.")
        else:
            # Apply filters
            filtered_papers = []
            for paper in papers:
                if require_eis and not paper["has_eis"]:
                    continue
                if require_ai and not paper["has_ai"]:
                    continue
                if paper["relevance"] < min_relevance:
                    continue
                filtered_papers.append(paper)
            
            if not filtered_papers:
                st.warning("No papers match your filters.")
            else:
                df = pd.DataFrame(filtered_papers)
                st.session_state.papers_df = df
                
                # Save to database
                db_file = init_metadata_db()
                if db_file:
                    conn = sqlite3.connect(db_file)
                    df.to_sql("papers", conn, if_exists="replace", index=False)
                    conn.close()
                
                # Display results
                st.success(f"Found **{len(df)}** papers")
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", len(df))
                with col2:
                    avg_rel = df['relevance'].mean()
                    st.metric("Avg Relevance", f"{avg_rel:.0f}")
                with col3:
                    eis_papers = df['has_eis'].sum()
                    st.metric("EIS Papers", eis_papers)
                with col4:
                    ai_papers = df['has_ai'].sum()
                    st.metric("AI Papers", ai_papers)
                
                # Display papers
                for idx, row in df.iterrows():
                    with st.expander(f"📄 **{row['title']}** ({row['year']}) | ⚡{row['relevance']}"):
                        st.markdown(f"**Authors**: {row['authors']}")
                        st.markdown(f"**Categories**: {row['categories']}")
                        
                        # Indicators
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if row['has_eis']:
                                st.success(f"✅ EIS (score: {row['eis_score']})")
                            else:
                                st.error("❌ No EIS")
                        with col2:
                            if row['has_ai']:
                                st.success(f"✅ AI/ML (score: {row['ai_score']})")
                            else:
                                st.error("❌ No AI")
                        with col3:
                            if st.button("📥 Download PDF", key=f"dl_{row['id']}"):
                                if download_pdf(row['pdf_url'], row['id']):
                                    st.success("Downloaded!")
                        
                        st.markdown("**Abstract**:")
                        st.markdown(f"{row['abstract'][:500]}..." if len(row['abstract']) > 500 else row['abstract'])
                        
                        # Links
                        abs_url = row['pdf_url'].replace('/pdf/', '/abs/')
                        st.markdown(f"[📖 arXiv Abstract]({abs_url}) | [📄 PDF]({row['pdf_url']})")

# ==============================
# DOWNLOAD SECTION
# ==============================
if st.session_state.papers_df is not None:
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    # Create download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV
        csv = st.session_state.papers_df.to_csv(index=False)
        st.download_button(
            "📋 CSV",
            csv,
            "eis_ai_papers.csv",
            "text/csv"
        )
    
    with col2:
        # JSON
        json_str = st.session_state.papers_df.to_json(orient="records", indent=2)
        st.download_button(
            "📄 JSON",
            json_str,
            "eis_ai_papers.json",
            "application/json"
        )
    
    with col3:
        # PDF ZIP
        if st.session_state.downloaded_pdfs:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for paper_id, pdf_bytes in st.session_state.downloaded_pdfs.items():
                    zipf.writestr(f"{paper_id}.pdf", pdf_bytes)
            
            st.download_button(
                "📦 PDFs (ZIP)",
                zip_buffer.getvalue(),
                "eis_ai_pdfs.zip",
                "application/zip"
            )
        else:
            st.button("📦 PDFs (ZIP)", disabled=True, help="Download some PDFs first")
    
    # Database download
    st.markdown("---")
    st.subheader("🗃️ Database")
    
    db_file = Path(tempfile.gettempdir()) / "eis_ai_metadata.db"
    if db_file.exists():
        with open(db_file, "rb") as f:
            db_bytes = f.read()
        
        st.download_button(
            "📊 Download SQLite DB",
            db_bytes,
            "eis_ai_metadata.db",
            "application/x-sqlite3"
        )
        
        # Show database preview
        with st.expander("🔍 View Database"):
            conn = sqlite3.connect(db_file)
            db_df = pd.read_sql("SELECT * FROM papers ORDER BY relevance DESC", conn)
            conn.close()
            
            st.dataframe(
                db_df[['title', 'year', 'relevance', 'eis_score', 'ai_score']],
                use_container_width=True
            )

# ==============================
# LOGS
# ==============================
st.markdown("---")
with st.expander("📝 Activity Logs"):
    if st.session_state.log_buffer:
        for log in st.session_state.log_buffer[-20:]:  # Show last 20 logs
            st.text(log)
    else:
        st.info("No logs yet. Perform a search to see activity.")

# ==============================
# TIPS SECTION
# ==============================
with st.sidebar:
    st.markdown("---")
    with st.expander("💡 Search Tips"):
        st.markdown("""
        **To get more papers:**
        1. Use "EIS AND AI/ML" for focused results
        2. Try "EIS only" or "AI/ML only" separately
        3. Lower the Min Relevance filter
        4. Uncheck "Require EIS" or "Require AI"
        5. Select multiple arXiv categories
        
        **Effective queries:**
        - `(EIS OR impedance) AND (machine learning OR neural)`
        - `electrochemical impedance AND battery AND ML`
        - `impedance spectroscopy AND data-driven`
        
        **Best categories:**
        - `physics.chem-ph` (Chemical Physics)
        - `cond-mat.mtrl-sci` (Materials Science)
        - `cs.LG` (Machine Learning)
        """)
