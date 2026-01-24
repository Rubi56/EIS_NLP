import arxiv
import pandas as pd
import streamlit as st
import sqlite3
import os
import gc
from datetime import datetime

# ==============================
# 1. DATABASE MANAGEMENT (The Fix)
# ==============================
# We store data in a temp file instead of RAM (Session State)
DB_PATH = "papers_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS papers
                 (id TEXT PRIMARY KEY, title TEXT, authors TEXT, 
                  year INTEGER, score INTEGER, abstract TEXT, 
                  pdf_url TEXT, arxiv_url TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(papers):
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame(papers)
    df.to_sql("papers", conn, if_exists="append", index=False, method="multi")
    conn.close()

def clear_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()

# ==============================
# 2. OPTIMIZED SEARCH
# ==============================
def simple_score(text):
    text_lower = text.lower()
    eis_terms = ['electrochemical impedance', 'eis', 'nyquist', 'bode']
    ai_terms = ['machine learning', 'deep learning', 'neural network', 'artificial intelligence']
    
    eis_count = sum(1 for term in eis_terms if term in text_lower)
    ai_count = sum(1 for term in ai_terms if term in text_lower)
    
    if eis_count > 0 and ai_count > 0:
        return 70 + (min(eis_count + ai_count, 5) * 6)
    return 0

def fetch_large_volume(query, max_results=500):
    client = arxiv.Client(page_size=100, delay_seconds=3.0)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.PublishedDate)
    
    batch = []
    count = 0
    
    for result in client.results(search):
        if result.published.year < 2010: continue
        
        score = simple_score(f"{result.title} {result.summary}")
        if score < 40: continue # Strict filter to save space
        
        batch.append({
            'id': result.get_short_id(),
            'title': result.title,
            'authors': ", ".join([a.name for a in result.authors[:2]]),
            'year': result.published.year,
            'score': score,
            'abstract': result.summary,
            'pdf_url': result.pdf_url,
            'arxiv_url': result.pdf_url.replace('/pdf/', '/abs/')
        })
        
        # Write to disk every 50 papers to keep RAM low
        if len(batch) >= 50:
            save_to_db(batch)
            batch = []
            gc.collect() # Force memory cleanup
        
        count += 1
    
    if batch:
        save_to_db(batch)
    return count

# ==============================
# 3. STREAMLIT UI
# ==============================
st.set_page_config(page_title="High-Volume EIS+AI Finder", layout="wide")
st.title("📚 High-Volume EIS & AI Research Explorer")
init_db()

with st.sidebar:
    st.header("🔍 Search Control")
    total_to_fetch = st.slider("Target Number of Papers", 100, 2000, 500)
    query_str = st.text_input("Query", '("electrochemical impedance spectroscopy" OR EIS) AND ("machine learning" OR AI)')
    
    if st.button("🚀 Run Deep Search", type="primary"):
        clear_db()
        with st.spinner("Fetching and indexing... this may take a minute."):
            found = fetch_large_volume(query_str, total_to_fetch)
            st.success(f"Indexed {found} relevant papers to disk!")

# Results Display (Paginated from Database)
conn = sqlite3.connect(DB_PATH)
try:
    # Check if we have data
    total_indexed = pd.read_sql("SELECT COUNT(*) FROM papers", conn).iloc[0,0]
    
    if total_indexed > 0:
        st.subheader(f"Results ({total_indexed} papers found since 2010)")
        
        # Export Option
        full_df = pd.read_sql("SELECT * FROM papers", conn)
        csv = full_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full CSV (Complete List)", csv, "eis_ai_library.csv", "text/csv")
        
        # Display Preview (only top 50 to save browser RAM)
        st.write("---")
        for _, row in full_df.head(50).iterrows():
            with st.expander(f"{row['year']} - {row['title']} (Score: {row['score']})"):
                st.write(f"**Authors:** {row['authors']}")
                st.write(row['abstract'])
                st.markdown(f"[PDF Link]({row['pdf_url']})")
                
        if total_indexed > 50:
            st.info(f"Showing first 50 results in UI. Download the CSV to see all {total_indexed} entries.")
    else:
        st.info("Use the sidebar to start a search.")
finally:
    conn.close()
