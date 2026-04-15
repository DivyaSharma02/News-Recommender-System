# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import time


# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsRadar — Smart News Finder",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --card:      #191d28;
    --border:    #252a38;
    --accent:    #f0c040;
    --accent2:   #e05c3a;
    --text:      #e8eaf0;
    --muted:     #6b7190;
    --radius:    12px;
}

/* ── Global ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }

/* ── Hero Header ── */
.hero {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 2.8rem 0 1.6rem;
}
.hero-icon {
    font-size: 3.4rem;
    line-height: 1;
    filter: drop-shadow(0 0 18px rgba(240,192,64,0.55));
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { filter: drop-shadow(0 0 14px rgba(240,192,64,0.45)); }
    50%       { filter: drop-shadow(0 0 28px rgba(240,192,64,0.85)); }
}
.hero-text h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.03em;
    background: linear-gradient(115deg, #f0c040 0%, #e05c3a 60%, #c93a8a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-text p {
    margin: 0.3rem 0 0;
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 40%, transparent 100%);
    margin: 0.5rem 0 2rem;
    opacity: 0.4;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.8rem 1.2rem !important; }
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.source-badge {
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    margin: 3px 2px;
    color: var(--muted);
}

/* ── Search Bar ── */
[data-testid="stTextInput"] > div > div > input {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(240,192,64,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div { accent-color: var(--accent); }

/* ── Metric Tiles ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-tile .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
}
.metric-tile .label {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 2px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── News Cards ── */
.news-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    transition: border-color 0.2s, transform 0.15s;
}
.news-card:hover {
    border-color: rgba(240,192,64,0.35);
    transform: translateX(3px);
}
.news-card .rank {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.news-card .title {
    font-size: 1.05rem;
    font-weight: 500;
    line-height: 1.45;
    margin-bottom: 0.7rem;
    color: var(--text);
}
.news-card .link-row {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.news-card .source-tag {
    font-size: 0.75rem;
    color: var(--muted);
    background: var(--surface);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
}
.news-card .score-bar-wrap {
    position: absolute;
    top: 1.4rem;
    right: 1.4rem;
    text-align: right;
}
.news-card .score-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--accent2);
}
.news-card .score-lbl {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Top-match highlight ── */
.news-card.top {
    border-color: rgba(240,192,64,0.5);
    background: linear-gradient(135deg, #1d2130 0%, var(--card) 100%);
}
.top-badge {
    display: inline-block;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #0d0f14;
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}
.empty-state .big {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.4rem;
}

/* ── Status pills ── */
.pill {
    display: inline-block;
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
}
.pill-ok  { background: rgba(50,200,100,0.15); color: #4cdb83; border: 1px solid rgba(50,200,100,0.3); }
.pill-warn{ background: rgba(240,192,64,0.15); color: var(--accent); border: 1px solid rgba(240,192,64,0.3); }

/* ── Progress bar override ── */
.stProgress > div > div > div { background-color: var(--accent) !important; }

/* ── Checkbox / multiselect ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── NLTK Setup ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = load_stopwords()

# ─── Constants ──────────────────────────────────────────────────────────────
SOURCES = {
    "Times of India": [
        "https://timesofindia.indiatimes.com/home/headlines",
        "https://timesofindia.indiatimes.com/india",
        "https://timesofindia.indiatimes.com/world",
        "https://timesofindia.indiatimes.com/business",
        "https://timesofindia.indiatimes.com/sports",
        "https://timesofindia.indiatimes.com/technology",
    ],
    "Hindustan Times": [
        "https://www.hindustantimes.com/india-news",
        "https://www.hindustantimes.com/world-news",
        "https://www.hindustantimes.com/business",
        "https://www.hindustantimes.com/tech",
        "https://www.hindustantimes.com/sports",
    ],
    "Indian Express": [
        "https://indianexpress.com/section/india",
        "https://indianexpress.com/section/world",
        "https://indianexpress.com/section/business",
        "https://indianexpress.com/section/technology",
        "https://indianexpress.com/section/sports",
    ],
    "NDTV": [
        "https://www.ndtv.com/india",
        "https://www.ndtv.com/world-news",
        "https://www.ndtv.com/business",
        "https://www.ndtv.com/technology",
        "https://www.ndtv.com/sports",
    ],
    "BBC": [
        "https://www.bbc.com/news",
        "https://www.bbc.com/news/world",
        "https://www.bbc.com/news/business",
        "https://www.bbc.com/news/technology",
        "https://www.bbc.com/sport",
    ],
}

HEADERS = {'User-Agent': 'Mozilla/5.0'}

SOURCE_DOMAIN_MAP = {
    "timesofindia": ("Times of India",  "https://timesofindia.indiatimes.com"),
    "hindustantimes": ("Hindustan Times","https://www.hindustantimes.com"),
    "indianexpress": ("Indian Express",  "https://indianexpress.com"),
    "ndtv":          ("NDTV",            "https://www.ndtv.com"),
    "bbc.com":       ("BBC",             "https://www.bbc.com"),
}

def get_source_label(url: str) -> str:
    for key, (label, _) in SOURCE_DOMAIN_MAP.items():
        if key in url:
            return label
    return "Unknown"

def paginate_url(base_url: str, page: int) -> str:
    if "timesofindia" in base_url:
        return f"{base_url}?page={page}"
    elif "hindustantimes" in base_url:
        return f"{base_url}/page-{page}"
    elif "indianexpress" in base_url:
        return f"{base_url}/page/{page}/"
    elif "ndtv" in base_url:
        return f"{base_url}/page-{page}"
    elif "bbc.com" in base_url:
        return f"{base_url}?page={page}"
    return base_url

def fix_relative_link(link: str, base_url: str) -> str:
    if link.startswith('/'):
        for key, (_, domain) in SOURCE_DOMAIN_MAP.items():
            if key in base_url:
                return domain + link
    return link

def scrape_section(url: str, pages: int = 2) -> list:
    articles = []
    for page in range(1, pages + 1):
        purl = paginate_url(url, page)
        try:
            resp = requests.get(purl, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            for a in soup.find_all('a', href=True):
                title = a.get_text(strip=True)
                if title and len(title) > 30:
                    link = fix_relative_link(a['href'], url)
                    articles.append({'title': title, 'link': link, 'source': get_source_label(url)})
        except Exception:
            pass
        time.sleep(0.5)
    return articles

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(w for w in text.split() if w not in stop_words)

def recommend_news(user_input: str, df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['cleaned_title'] = df['title'].apply(clean_text)
    query_clean = clean_text(user_input)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_title'])
    query_vec = vectorizer.transform([query_clean])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df['score'] = scores
    return df[df['score'] > 0].sort_values('score', ascending=False).head(top_n)


# ─── Session State ───────────────────────────────────────────────────────────
if 'news_df' not in st.session_state:
    st.session_state.news_df = pd.DataFrame()
if 'scraped' not in st.session_state:
    st.session_state.scraped = False

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📡 NewsRadar</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Select Sources**")
    selected_sources = st.multiselect(
        label="Sources",
        options=list(SOURCES.keys()),
        default=list(SOURCES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("**Pages per Section**")
    pages_per_section = st.slider("Pages", 1, 5, 2, label_visibility="collapsed")

    st.markdown("**Results to Show**")
    top_n = st.slider("Top N", 3, 20, 8, label_visibility="collapsed")

    st.markdown("")
    fetch_btn = st.button("🔄 Fetch Latest News", use_container_width=True)

    if st.session_state.scraped:
        st.markdown("")
        st.markdown(
            f'<span class="pill pill-ok">✓ {len(st.session_state.news_df)} articles loaded</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#6b7190;line-height:1.6;">'
                'TF-IDF + cosine similarity engine.<br>Scrapes live headlines from major Indian & international news portals.'
                '</div>', unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">📡</div>
  <div class="hero-text">
    <h1>NewsRadar</h1>
    <p>Smart semantic news discovery — find stories that match what you care about</p>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ─── Fetch Logic ─────────────────────────────────────────────────────────────
if fetch_btn:
    if not selected_sources:
        st.warning("Please select at least one news source.")
    else:
        urls = [u for src in selected_sources for u in SOURCES[src]]
        all_articles = []

        progress_bar = st.progress(0, text="Fetching headlines…")
        status_text  = st.empty()

        for i, url in enumerate(urls):
            label = get_source_label(url)
            status_text.markdown(f'<span class="pill pill-warn">⏳ Scraping {label}…</span>', unsafe_allow_html=True)
            all_articles.extend(scrape_section(url, pages=pages_per_section))
            progress_bar.progress((i + 1) / len(urls), text=f"Scraped {i+1}/{len(urls)} sections")

        progress_bar.empty()
        status_text.empty()

        if all_articles:
            df = pd.DataFrame(all_articles).drop_duplicates(subset=['title'])
            df = df[df['title'].str.len() > 30].reset_index(drop=True)
            st.session_state.news_df = df
            st.session_state.scraped = True
            st.success(f"✅ Indexed **{len(df)}** unique articles from {len(selected_sources)} source(s).")
        else:
            st.error("Could not fetch any articles. Check your internet connection.")


# ─── Metrics Row ─────────────────────────────────────────────────────────────
if st.session_state.scraped and not st.session_state.news_df.empty:
    df = st.session_state.news_df
    src_counts = df['source'].value_counts()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""<div class="metric-tile">
            <div class="value">{len(df)}</div>
            <div class="label">Articles Indexed</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="metric-tile">
            <div class="value">{df['source'].nunique()}</div>
            <div class="label">News Sources</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        top_src = src_counts.idxmax() if not src_counts.empty else "—"
        st.markdown(f"""<div class="metric-tile">
            <div class="value" style="font-size:1.1rem;padding-top:0.4rem">{top_src}</div>
            <div class="label">Most Articles From</div>
        </div>""", unsafe_allow_html=True)


# ─── Search Bar ──────────────────────────────────────────────────────────────
st.markdown("### 🔍 What are you interested in?")
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        label="Search topic",
        placeholder="e.g.  AI breakthroughs  ·  India elections  ·  cricket IPL  ·  stock market",
        label_visibility="collapsed",
    )
with col2:
    search_btn = st.button("Search →", use_container_width=True)


# ─── Results ─────────────────────────────────────────────────────────────────
if search_btn or query:
    if not st.session_state.scraped or st.session_state.news_df.empty:
        st.markdown("""<div class="empty-state">
            <div class="big">📭</div>
            <h3>No articles loaded yet</h3>
            <p>Click <b>Fetch Latest News</b> in the sidebar to scrape headlines first.</p>
        </div>""", unsafe_allow_html=True)
    elif not query.strip():
        st.info("Type a topic above and hit **Search →**")
    else:
        results = recommend_news(query, st.session_state.news_df, top_n=top_n)

        if results.empty:
            st.markdown(f"""<div class="empty-state">
                <div class="big">🔭</div>
                <h3>No matches found for "{query}"</h3>
                <p>Try broader keywords or fetch fresh articles.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(results)} results** for &nbsp;`{query}`", unsafe_allow_html=True)
            st.markdown("")

            for i, row in enumerate(results.itertuples(), 1):
                is_top = i == 1
                card_class = "news-card top" if is_top else "news-card"
                top_badge  = '<span class="top-badge">Top Match</span>' if is_top else ""
                score_pct  = f"{row.score * 100:.0f}%"
                source     = getattr(row, 'source', 'Unknown')

                st.markdown(f"""
                <div class="{card_class}">
                    <div class="score-bar-wrap">
                        <div class="score-num">{score_pct}</div>
                        <div class="score-lbl">Match</div>
                    </div>
                    <div class="rank">#{i}{top_badge}</div>
                    <div class="title">{row.title}</div>
                    <div class="link-row">
                        <span class="source-tag">{source}</span>
                        <a href="{row.link}" target="_blank"
                           style="font-size:0.82rem;color:#f0c040;text-decoration:none;">
                            Read Article ↗
                        </a>
                    </div>
                </div>""", unsafe_allow_html=True)

elif not st.session_state.scraped:
    st.markdown("""<div class="empty-state">
        <div class="big">🗞️</div>
        <h3>Start by fetching news</h3>
        <p>Use the sidebar to select sources and click <b>Fetch Latest News</b>.<br>
        Then search for any topic you care about.</p>
    </div>""", unsafe_allow_html=True)
