import streamlit as st
import streamlit.components.v1 as components
import sys
import os
sys.path.append('backend')

from backend.src.utils.database import execute_query, init_db
from backend.src.services.rss_scraper import scrape_all_feeds
from backend.src.services.stock_fetcher import update_all_stocks
from backend.config import Config

# Page config
st.set_page_config(
    page_title="Quantum Tracker",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit branding
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize backend
config = Config()
init_db(config.DATABASE_PATH)

# Check and initialize data if needed
@st.cache_data(ttl=60)
def ensure_data():
    try:
        news_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM news_articles")[0]['count']
        stock_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM stock_data")[0]['count']
        
        # Initialize if empty
        if news_count == 0:
            scrape_all_feeds(config.__dict__)
        if stock_count == 0:
            update_all_stocks(config.__dict__)
        
        return True
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return False

# Ensure data exists
ensure_data()

# Serve the React frontend
react_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Tracker</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="module">
        // Your React app will be loaded here
        const API_URL = window.location.origin;
        
        // Fetch and render your frontend
        fetch('/app/static/index.html')
            .then(response => response.text())
            .then(html => {
                document.getElementById('root').innerHTML = html;
            });
    </script>
</body>
</html>
"""

# Embed the React frontend
components.html(react_html, height=1000, scrolling=True)
