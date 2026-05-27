"""
RecallBot - AI Memory Assistant for Team Chats
Production-grade Streamlit app with ChromaDB, OpenAI, and beautiful UI
"""

import streamlit as st
import os
import json
import re
import zipfile
import hashlib
import sqlite3
import time
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO, BytesIO
from typing import Optional
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

from dotenv import load_dotenv

load_dotenv()

# ─── PAGE CONFIG (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="RecallBot · AI Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LAZY IMPORTS (avoid crash if deps missing) ───────────────────────────────
try:
    import openai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LC_OK = True
except ImportError:
    LC_OK = False

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DB_PATH = "recallbot.db"
CHROMA_PATH = "./chroma_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
GPT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
MAX_RESULTS = 6

# ─── CSS INJECTION ────────────────────────────────────────────────────────────
def inject_css():
    theme = st.session_state.get("theme", "dark")
    if theme == "dark":
        bg = "#0d0f14"
        card = "#161a24"
        card2 = "#1e2333"
        border = "#2a3040"
        text = "#e8eaf0"
        sub = "#8892a4"
        accent = "#6c63ff"
        accent2 = "#00d4aa"
        accent3 = "#ff6b6b"
        inp_bg = "#1a1f2e"
    else:
        bg = "#f0f2f8"
        card = "#ffffff"
        card2 = "#f8f9fc"
        border = "#dde2ee"
        text = "#1a1f2e"
        sub = "#6b7280"
        accent = "#6c63ff"
        accent2 = "#00b894"
        accent3 = "#e17055"
        inp_bg = "#ffffff"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── ROOT ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, .stApp {{
    background: {bg} !important;
    color: {text} !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: {card} !important;
    border-right: 1px solid {border} !important;
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
}}

/* ── MAIN CONTAINER ── */
.main .block-container {{
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1200px !important;
}}

/* ── BRAND HEADER ── */
.rb-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 18px 20px 14px;
    border-bottom: 1px solid {border};
    margin-bottom: 8px;
}}
.rb-brand-icon {{
    width: 36px; height: 36px;
    background: linear-gradient(135deg, {accent}, {accent2});
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 15px {accent}44;
}}
.rb-brand-name {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    background: linear-gradient(135deg, {accent}, {accent2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* ── NAV BUTTONS ── */
.rb-nav-btn {{
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 10px 16px;
    border: none;
    border-radius: 10px;
    background: transparent;
    color: {sub};
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 2px;
    text-align: left;
}}
.rb-nav-btn:hover {{
    background: {card2};
    color: {text};
}}
.rb-nav-btn.active {{
    background: linear-gradient(135deg, {accent}22, {accent2}11);
    color: {text};
    border-left: 3px solid {accent};
}}

/* ── PAGE TITLE ── */
.rb-page-title {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(135deg, {text} 0%, {sub} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}}
.rb-page-sub {{
    color: {sub};
    font-size: 0.88rem;
    margin-bottom: 1.5rem;
}}

/* ── CARDS ── */
.rb-card {{
    background: {card};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    transition: box-shadow 0.2s;
}}
.rb-card:hover {{
    box-shadow: 0 8px 30px {accent}15;
}}
.rb-card-sm {{
    background: {card};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}

/* ── METRIC CARDS ── */
.rb-metric {{
    background: {card};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
}}
.rb-metric-val {{
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, {accent}, {accent2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}}
.rb-metric-label {{
    font-size: 0.78rem;
    color: {sub};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ── CHAT BUBBLES ── */
.rb-chat-wrap {{
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 8px 0;
}}
.rb-msg-user {{
    display: flex;
    justify-content: flex-end;
}}
.rb-msg-user .rb-bubble {{
    background: linear-gradient(135deg, {accent}, {accent}dd);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    max-width: 70%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px {accent}44;
}}
.rb-msg-ai {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
}}
.rb-ai-avatar {{
    width: 32px; height: 32px;
    background: linear-gradient(135deg, {accent2}, {accent});
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px {accent2}44;
}}
.rb-msg-ai .rb-bubble {{
    background: {card2};
    border: 1px solid {border};
    color: {text};
    border-radius: 4px 18px 18px 18px;
    padding: 12px 16px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
}}

/* ── CITATION CHIP ── */
.rb-citation {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: {card2};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 0.78rem;
    color: {sub};
    margin: 4px 4px 4px 0;
}}
.rb-citation-sender {{
    color: {accent2};
    font-weight: 600;
}}
.rb-citation-time {{
    color: {accent};
    font-size: 0.72rem;
}}

/* ── BADGE ── */
.rb-badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
}}
.rb-badge-purple {{ background: {accent}22; color: {accent}; }}
.rb-badge-green  {{ background: {accent2}22; color: {accent2}; }}
.rb-badge-red    {{ background: {accent3}22; color: {accent3}; }}

/* ── INPUT BOX ── */
.stTextInput input, .stTextArea textarea {{
    background: {inp_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 12px !important;
    color: {text} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {accent} !important;
    box-shadow: 0 0 0 3px {accent}22 !important;
}}

/* ── BUTTONS ── */
.stButton > button {{
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {accent}, {accent}cc) !important;
    border: none !important;
    box-shadow: 0 4px 15px {accent}44 !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px {accent}55 !important;
}}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploadDropzone"] {{
    background: {card2} !important;
    border: 2px dashed {border} !important;
    border-radius: 16px !important;
    transition: all 0.2s !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {accent} !important;
    background: {accent}08 !important;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {card} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid {border} !important;
    gap: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {accent}, {accent}cc) !important;
    color: #fff !important;
}}

/* ── PROGRESS BAR ── */
.rb-progress-bar {{
    height: 6px;
    background: {border};
    border-radius: 100px;
    overflow: hidden;
    margin: 6px 0;
}}
.rb-progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, {accent}, {accent2});
    border-radius: 100px;
    transition: width 0.4s ease;
}}

/* ── DIVIDER ── */
.rb-divider {{
    height: 1px;
    background: {border};
    margin: 12px 0;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: {sub}; }}

/* ── SELECT / MULTISELECT ── */
[data-baseweb="select"] > div {{
    background: {inp_bg} !important;
    border-color: {border} !important;
    border-radius: 10px !important;
}}

/* ── ALERT BOXES ── */
.rb-alert {{
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 0.88rem;
    margin-bottom: 12px;
}}
.rb-alert-info  {{ background: {accent}15; border-left: 3px solid {accent}; color: {text}; }}
.rb-alert-ok    {{ background: {accent2}15; border-left: 3px solid {accent2}; color: {text}; }}
.rb-alert-warn  {{ background: {accent3}15; border-left: 3px solid {accent3}; color: {text}; }}

/* ── ANIMATIONS ── */
@keyframes fadeIn {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:none; }} }}
@keyframes pulse  {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.5; }} }}
.rb-fade {{ animation: fadeIn 0.35s ease both; }}
.rb-pulse {{ animation: pulse 1.5s ease infinite; }}

/* ── TYPING DOTS ── */
.rb-typing {{ display:flex; gap:5px; align-items:center; padding:6px 0; }}
.rb-typing span {{
    width:7px; height:7px;
    background: {accent};
    border-radius:50%;
    animation: pulse 1.2s ease infinite;
}}
.rb-typing span:nth-child(2) {{ animation-delay:0.2s; }}
.rb-typing span:nth-child(3) {{ animation-delay:0.4s; }}

/* ── STORAGE METER ── */
.rb-storage-bar {{
    height:4px; background:{border}; border-radius:100px; margin-top:4px;
}}
.rb-storage-fill {{
    height:100%;
    background:linear-gradient(90deg,{accent},{accent2});
    border-radius:100px;
}}
</style>
""", unsafe_allow_html=True)

# ─── DATABASE SETUP ───────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        theme TEXT DEFAULT 'dark'
    );
    CREATE TABLE IF NOT EXISTS chat_collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        source_type TEXT NOT NULL,
        message_count INTEGER DEFAULT 0,
        indexed_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS messages_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id INTEGER NOT NULL,
        sender TEXT,
        timestamp TEXT,
        content TEXT NOT NULL,
        FOREIGN KEY (collection_id) REFERENCES chat_collections(id)
    );
    CREATE TABLE IF NOT EXISTS saved_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        query TEXT NOT NULL,
        answer TEXT,
        saved_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS bookmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        message_id INTEGER NOT NULL,
        note TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (message_id) REFERENCES messages_raw(id)
    );
    """)
    conn.commit()
    conn.close()

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def create_user(username, email, password):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
            (username, email, hash_password(password))
        )
        conn.commit()
        return True, "Account created!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, hash_password(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_collections(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM chat_collections WHERE user_id=? ORDER BY indexed_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_messages(collection_id, limit=None):
    conn = get_db()
    q = "SELECT * FROM messages_raw WHERE collection_id=? ORDER BY timestamp"
    args = [collection_id]
    if limit:
        q += " LIMIT ?"
        args.append(limit)
    rows = conn.execute(q, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_query(user_id, query, answer):
    conn = get_db()
    conn.execute(
        "INSERT INTO saved_queries (user_id, query, answer) VALUES (?,?,?)",
        (user_id, query, answer)
    )
    conn.commit()
    conn.close()

def get_saved_queries(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM saved_queries WHERE user_id=? ORDER BY saved_at DESC LIMIT 20",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ─── PARSERS ──────────────────────────────────────────────────────────────────
WA_PATTERNS = [
    r"(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})[,\s]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\s*[-–]\s*([^:]+?):\s(.+)",
    r"\[(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]\s([^:]+?):\s(.+)",
]
MEDIA_SKIP = {"<media omitted>", "<image omitted>", "<video omitted>", "<audio omitted>", "<sticker omitted>"}

def parse_whatsapp(text: str) -> list[dict]:
    messages = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower() in MEDIA_SKIP:
            continue
        if any(skip in line.lower() for skip in ["omitted", "end-to-end encrypted"]):
            continue
        for pat in WA_PATTERNS:
            m = re.match(pat, line)
            if m:
                groups = m.groups()
                date_str, time_str, sender, content = groups[0], groups[1], groups[2].strip(), groups[3].strip()
                ts = f"{date_str} {time_str}"
                messages.append({"sender": sender, "timestamp": ts, "content": content})
                break
    return messages

def parse_slack_json(data) -> list[dict]:
    messages = []
    if isinstance(data, list):
        for msg in data:
            if msg.get("type") != "message" or msg.get("subtype"):
                continue
            ts_raw = msg.get("ts", "")
            try:
                ts = datetime.fromtimestamp(float(ts_raw)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = ts_raw
            sender = msg.get("user", msg.get("username", "unknown"))
            text = msg.get("text", "").strip()
            if text:
                messages.append({"sender": sender, "timestamp": ts, "content": text})
    return messages

def parse_slack_zip(zip_bytes: bytes) -> list[dict]:
    messages = []
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
            for name in z.namelist():
                if name.endswith(".json") and "/" in name:
                    with z.open(name) as f:
                        try:
                            data = json.load(f)
                            messages.extend(parse_slack_json(data))
                        except Exception:
                            pass
    except Exception:
        pass
    return messages

# ─── VECTOR ENGINE ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embed_model():
    if not ST_OK:
        return None
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    if not CHROMA_OK:
        return None
    Path(CHROMA_PATH).mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

def collection_name(user_id, coll_id):
    return f"u{user_id}_c{coll_id}"

def index_messages(user_id, coll_id, messages: list[dict]):
    client = get_chroma_client()
    embed = get_embed_model()
    if not client or not embed:
        return False

    cname = collection_name(user_id, coll_id)
    try:
        client.delete_collection(cname)
    except Exception:
        pass
    coll = client.create_collection(cname)

    if LC_OK:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    else:
        splitter = None

    docs, metas, ids = [], [], []
    for i, msg in enumerate(messages):
        text = msg["content"]
        chunks = splitter.split_text(text) if splitter else [text]
        for j, chunk in enumerate(chunks):
            docs.append(chunk)
            metas.append({
                "sender": msg.get("sender", ""),
                "timestamp": msg.get("timestamp", ""),
                "msg_id": i,
                "collection_id": coll_id,
            })
            ids.append(f"m{i}_c{j}")

    if not docs:
        return False

    batch_size = 100
    for start in range(0, len(docs), batch_size):
        batch_docs = docs[start:start+batch_size]
        batch_metas = metas[start:start+batch_size]
        batch_ids = ids[start:start+batch_size]
        embeddings = embed.encode(batch_docs, show_progress_bar=False).tolist()
        coll.add(documents=batch_docs, embeddings=embeddings, metadatas=batch_metas, ids=batch_ids)
    return True

def semantic_search(user_id, coll_id, query: str, n=MAX_RESULTS):
    client = get_chroma_client()
    embed = get_embed_model()
    if not client or not embed:
        return []
    try:
        coll = client.get_collection(collection_name(user_id, coll_id))
        q_emb = embed.encode([query]).tolist()
        results = coll.query(query_embeddings=q_emb, n_results=min(n, coll.count()))
        out = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            out.append({"content": doc, "sender": meta.get("sender",""), "timestamp": meta.get("timestamp","")})
        return out
    except Exception:
        return []

# ─── AI ANSWER ────────────────────────────────────────────────────────────────
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("openai_key", "")
    if not api_key or not OPENAI_OK:
        return None
    return openai.OpenAI(api_key=api_key)

def build_context(hits: list[dict]) -> str:
    ctx = ""
    for i, h in enumerate(hits):
        ctx += f"[{i+1}] [{h['timestamp']}] {h['sender']}: {h['content']}\n"
    return ctx

def ask_ai(question: str, context: str):
    client = get_openai_client()
    if not client:
        yield "⚠️ OpenAI API key not set. Please add it in Settings.", []
        return

    system = (
        "You are RecallBot, an AI assistant that answers questions about team chat history. "
        "Use ONLY the provided context messages to answer. "
        "Always cite your sources using [N] notation matching the context. "
        "If the answer is not in the context, say so clearly. "
        "Be concise but thorough. Format well using markdown."
    )
    user_msg = f"Context messages:\n{context}\n\nQuestion: {question}"
    try:
        stream = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            stream=True,
            max_tokens=700,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            yield delta
    except Exception as e:
        yield f"⚠️ API Error: {e}"

# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
def init_session():
    defaults = {
        "user": None,
        "page": "chat",
        "theme": "dark",
        "chat_history": [],
        "active_collection": None,
        "openai_key": os.environ.get("OPENAI_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─── AUTH PAGE ────────────────────────────────────────────────────────────────
def page_auth():
    st.markdown("""
<div style="display:flex;justify-content:center;align-items:center;min-height:80vh;">
  <div style="width:100%;max-width:420px;">
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
<div style="text-align:center;margin-bottom:2rem;">
  <div style="font-size:3rem;margin-bottom:12px;">🧠</div>
  <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;
    background:linear-gradient(135deg,#6c63ff,#00d4aa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    RecallBot
  </div>
  <div style="color:#8892a4;font-size:0.9rem;margin-top:6px;">
    AI Memory for Your Team Chats
  </div>
</div>
""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Sign In", "✨ Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="your_username")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
            if submitted:
                user = verify_user(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.theme = user.get("theme", "dark")
                    st.success(f"Welcome back, {username}! 👋")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        st.markdown("""<div style="text-align:center;margin-top:8px;font-size:0.8rem;color:#8892a4;">
        Demo: username <b>demo</b> · password <b>demo123</b></div>""", unsafe_allow_html=True)

    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("Username", placeholder="choose_username")
            new_email = st.text_input("Email", placeholder="you@company.com")
            new_pwd = st.text_input("Password", type="password", placeholder="min 6 chars")
            new_pwd2 = st.text_input("Confirm Password", type="password", placeholder="repeat password")
            sub2 = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if sub2:
                if not new_user or not new_email or not new_pwd:
                    st.error("All fields required.")
                elif new_pwd != new_pwd2:
                    st.error("Passwords don't match.")
                elif len(new_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, msg = create_user(new_user, new_email, new_pwd)
                    if ok:
                        st.success("Account created! Please sign in.")
                    else:
                        st.error(msg)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar():
    user = st.session_state.user
    colls = get_user_collections(user["id"])

    st.sidebar.markdown(f"""
<div class="rb-brand">
  <div class="rb-brand-icon">🧠</div>
  <div class="rb-brand-name">RecallBot</div>
</div>
""", unsafe_allow_html=True)

    # Nav
    pages = [
        ("chat",     "💬", "Ask AI"),
        ("upload",   "📤", "Import Chats"),
        ("search",   "🔍", "Search"),
        ("analytics","📊", "Analytics"),
        ("memory",   "📌", "Memory"),
        ("settings", "⚙️", "Settings"),
    ]
    for page_id, icon, label in pages:
        active = "active" if st.session_state.page == page_id else ""
        if st.sidebar.button(f"{icon}  {label}", key=f"nav_{page_id}", use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

    st.sidebar.markdown('<div class="rb-divider"></div>', unsafe_allow_html=True)

    # Active collection picker
    if colls:
        st.sidebar.markdown('<div style="padding:0 4px;font-size:0.75rem;color:#8892a4;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Active Collection</div>', unsafe_allow_html=True)
        coll_options = {c["name"]: c["id"] for c in colls}
        chosen_name = st.sidebar.selectbox("", list(coll_options.keys()), label_visibility="collapsed")
        st.session_state.active_collection = coll_options[chosen_name]

    st.sidebar.markdown('<div class="rb-divider"></div>', unsafe_allow_html=True)

    # Storage meter
    conn = get_db()
    total_msgs = conn.execute("SELECT COUNT(*) FROM messages_raw WHERE collection_id IN (SELECT id FROM chat_collections WHERE user_id=?)", (user["id"],)).fetchone()[0]
    conn.close()
    used_pct = min(100, int(total_msgs / 50))
    st.sidebar.markdown(f"""
<div style="padding:0 4px;">
  <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8892a4;">
    <span>💾 Storage</span><span>{total_msgs} messages</span>
  </div>
  <div class="rb-storage-bar">
    <div class="rb-storage-fill" style="width:{used_pct}%"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="rb-divider"></div>', unsafe_allow_html=True)

    # User + logout
    st.sidebar.markdown(f"""
<div style="padding:8px 4px;display:flex;align-items:center;gap:8px;">
  <div style="width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#6c63ff,#00d4aa);
    display:flex;align-items:center;justify-content:center;font-size:12px;color:#fff;font-weight:700;">
    {user['username'][0].upper()}
  </div>
  <div>
    <div style="font-size:0.85rem;font-weight:600;">{user['username']}</div>
    <div style="font-size:0.72rem;color:#8892a4;">{user['email']}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    if st.sidebar.button("🚪  Sign Out", use_container_width=True):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.rerun()

# ─── CHAT PAGE ────────────────────────────────────────────────────────────────
def page_chat():
    st.markdown('<div class="rb-page-title">Ask AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="rb-page-sub">Ask anything about your imported chats — AI will cite exact messages.</div>', unsafe_allow_html=True)

    user = st.session_state.user
    coll_id = st.session_state.active_collection

    if not coll_id:
        st.markdown('<div class="rb-alert rb-alert-info">📤 No collection selected. Go to <b>Import Chats</b> to upload your first chat file.</div>', unsafe_allow_html=True)
        return

    if not st.session_state.get("openai_key"):
        st.markdown('<div class="rb-alert rb-alert-warn">🔑 OpenAI API key missing. Add it in <b>Settings</b>.</div>', unsafe_allow_html=True)

    # Chat history
    history = st.session_state.chat_history
    chat_html = '<div class="rb-chat-wrap">'
    for turn in history:
        if turn["role"] == "user":
            chat_html += f'<div class="rb-msg-user rb-fade"><div class="rb-bubble">{turn["content"]}</div></div>'
        else:
            chat_html += f'<div class="rb-msg-ai rb-fade"><div class="rb-ai-avatar">🤖</div><div class="rb-bubble">{turn["content"]}</div></div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Suggested questions
    if not history:
        st.markdown('<div style="margin:8px 0 4px;font-size:0.8rem;color:#8892a4;">Try asking:</div>', unsafe_allow_html=True)
        suggestions = [
            "Summarize the key decisions made",
            "Who mentioned the deadline?",
            "Find all action items",
            "What was discussed last week?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state["prefill_question"] = s
                    st.rerun()

    # Input
    prefill = st.session_state.pop("prefill_question", "")
    question = st.chat_input("Ask anything about your chats…")
    if not question and prefill:
        question = prefill

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="rb-msg-user rb-fade"><div class="rb-bubble">{question}</div></div>', unsafe_allow_html=True)

        # Search
        with st.spinner(""):
            hits = semantic_search(user["id"], coll_id, question)

        if not hits:
            answer = "No relevant messages found in the current collection."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.markdown(f'<div class="rb-msg-ai rb-fade"><div class="rb-ai-avatar">🤖</div><div class="rb-bubble">{answer}</div></div>', unsafe_allow_html=True)
            return

        context = build_context(hits)

        # Stream response
        answer_placeholder = st.empty()
        full_answer = ""
        with answer_placeholder.container():
            st.markdown('<div class="rb-msg-ai rb-fade"><div class="rb-ai-avatar">🤖</div><div class="rb-bubble"><div class="rb-typing"><span></span><span></span><span></span></div></div></div>', unsafe_allow_html=True)

        full_answer = ""
        for chunk in ask_ai(question, context):
            full_answer += chunk
            answer_placeholder.markdown(f'<div class="rb-msg-ai rb-fade"><div class="rb-ai-avatar">🤖</div><div class="rb-bubble">{full_answer}▌</div></div>', unsafe_allow_html=True)

        answer_placeholder.markdown(f'<div class="rb-msg-ai rb-fade"><div class="rb-ai-avatar">🤖</div><div class="rb-bubble">{full_answer}</div></div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})

        # Citations
        st.markdown('<div style="margin-top:8px;"><div style="font-size:0.75rem;color:#8892a4;margin-bottom:4px;">📎 Source messages</div>', unsafe_allow_html=True)
        cit_html = ""
        for i, h in enumerate(hits):
            cit_html += f'<span class="rb-citation"><span class="rb-citation-sender">[{i+1}] {h["sender"]}</span><span class="rb-citation-time">{h["timestamp"]}</span> · {h["content"][:60]}…</span>'
        st.markdown(cit_html + "</div>", unsafe_allow_html=True)

        # Save option
        if st.button("📌 Save this Q&A", key="save_qa"):
            save_query(user["id"], question, full_answer)
            st.success("Saved!")

    # Clear chat
    if history and st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─── UPLOAD PAGE ──────────────────────────────────────────────────────────────
def page_upload():
    st.markdown('<div class="rb-page-title">Import Chats</div>', unsafe_allow_html=True)
    st.markdown('<div class="rb-page-sub">Upload WhatsApp exports (.txt) or Slack exports (.json / .zip)</div>', unsafe_allow_html=True)

    user = st.session_state.user

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 Upload File")

        source_type = st.radio("Source type", ["WhatsApp (.txt)", "Slack (.json)", "Slack Export (.zip)"], horizontal=True)
        coll_name = st.text_input("Collection name", placeholder="e.g. Engineering Team 2024")
        uploaded = st.file_uploader(
            "Drop your file here",
            type=["txt", "json", "zip"],
            help="WhatsApp: export chat as .txt | Slack: export workspace or channel"
        )

        if uploaded and coll_name:
            if st.button("⚡ Index Chats", type="primary", use_container_width=True):
                with st.status("Processing your file…", expanded=True) as status:
                    st.write("📖 Parsing messages…")
                    try:
                        if "WhatsApp" in source_type:
                            text = uploaded.read().decode("utf-8", errors="ignore")
                            messages = parse_whatsapp(text)
                        elif source_type == "Slack (.json)":
                            data = json.load(uploaded)
                            messages = parse_slack_json(data)
                        else:
                            messages = parse_slack_zip(uploaded.read())
                    except Exception as e:
                        st.error(f"Parse error: {e}")
                        messages = []

                    if not messages:
                        st.error("No messages found. Check the file format.")
                        status.update(label="Failed", state="error")
                    else:
                        st.write(f"✅ Parsed {len(messages)} messages")
                        st.write("💾 Saving to database…")

                        conn = get_db()
                        cur = conn.cursor()
                        cur.execute(
                            "INSERT INTO chat_collections (user_id, name, source_type, message_count) VALUES (?,?,?,?)",
                            (user["id"], coll_name, source_type, len(messages))
                        )
                        coll_id = cur.lastrowid
                        for msg in messages:
                            cur.execute(
                                "INSERT INTO messages_raw (collection_id, sender, timestamp, content) VALUES (?,?,?,?)",
                                (coll_id, msg.get("sender",""), msg.get("timestamp",""), msg["content"])
                            )
                        conn.commit()
                        conn.close()

                        st.write("🔢 Generating embeddings…")
                        if CHROMA_OK and ST_OK:
                            ok = index_messages(user["id"], coll_id, messages)
                            if ok:
                                st.write("✅ Vector index built")
                            else:
                                st.write("⚠️ Indexing skipped (vector deps unavailable)")
                        else:
                            st.write("⚠️ ChromaDB/SentenceTransformers not available — search will be disabled")

                        status.update(label=f"Done! {len(messages)} messages indexed.", state="complete")
                        st.session_state.active_collection = coll_id
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        st.markdown("#### 📚 Your Collections")
        colls = get_user_collections(user["id"])
        if not colls:
            st.markdown('<div style="color:#8892a4;font-size:0.88rem;">No collections yet.</div>', unsafe_allow_html=True)
        for c in colls:
            active = c["id"] == st.session_state.active_collection
            badge = "🟢" if active else "⚪"
            st.markdown(f"""
<div class="rb-card-sm" style="{'border-color:#6c63ff;' if active else ''}">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div style="font-weight:600;font-size:0.9rem;">{badge} {c['name']}</div>
      <div style="font-size:0.75rem;color:#8892a4;">{c['message_count']} msgs · {c['source_type']}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            if not active:
                if st.button("Activate", key=f"act_{c['id']}"):
                    st.session_state.active_collection = c["id"]
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Format guides
        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        st.markdown("#### 📖 Format Guide")
        st.markdown("""
**WhatsApp**: Go to chat → ⋮ → More → Export Chat → Without Media → save .txt

**Slack**: Workspace Settings → Import/Export → Export → All time → Download .zip
""")
        st.markdown('</div>', unsafe_allow_html=True)

# ─── SEARCH PAGE ──────────────────────────────────────────────────────────────
def page_search():
    st.markdown('<div class="rb-page-title">Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="rb-page-sub">Semantic and keyword search across your chat history</div>', unsafe_allow_html=True)

    user = st.session_state.user
    coll_id = st.session_state.active_collection

    if not coll_id:
        st.markdown('<div class="rb-alert rb-alert-info">Select a collection first.</div>', unsafe_allow_html=True)
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("🔍 Search query", placeholder="What were the action items?")
    with col2:
        mode = st.selectbox("Mode", ["Semantic (AI)", "Keyword"])

    # Date + user filters
    c1, c2, c3 = st.columns(3)
    with c1:
        date_from = st.date_input("From date", value=None)
    with c2:
        date_to = st.date_input("To date", value=None)
    with c3:
        conn = get_db()
        senders = conn.execute("SELECT DISTINCT sender FROM messages_raw WHERE collection_id=?", (coll_id,)).fetchall()
        conn.close()
        sender_list = [r[0] for r in senders if r[0]]
        filter_sender = st.multiselect("Filter by sender", sender_list)

    if query:
        if mode == "Semantic (AI)":
            results = semantic_search(user["id"], coll_id, query)
            # Apply filters
            if date_from:
                results = [r for r in results if r.get("timestamp","") >= str(date_from)]
            if date_to:
                results = [r for r in results if r.get("timestamp","") <= str(date_to)]
            if filter_sender:
                results = [r for r in results if r.get("sender","") in filter_sender]
        else:
            conn = get_db()
            sql = "SELECT sender, timestamp, content FROM messages_raw WHERE collection_id=? AND content LIKE ?"
            args = [coll_id, f"%{query}%"]
            if filter_sender:
                placeholders = ",".join("?" for _ in filter_sender)
                sql += f" AND sender IN ({placeholders})"
                args.extend(filter_sender)
            rows = conn.execute(sql, args).fetchall()
            conn.close()
            results = [{"sender": r[0], "timestamp": r[1], "content": r[2]} for r in rows[:20]]

        st.markdown(f'<div style="margin:8px 0;font-size:0.82rem;color:#8892a4;">{len(results)} result(s)</div>', unsafe_allow_html=True)

        if results:
            if PANDAS_OK:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, height=400)
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Export CSV", csv, "recallbot_search.csv", "text/csv")
            else:
                for r in results:
                    st.markdown(f"**{r.get('sender','')}** · {r.get('timestamp','')}

{r.get('content','')}")

# ─── ANALYTICS PAGE ───────────────────────────────────────────────────────────
def page_analytics():
    st.markdown('<div class="rb-page-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="rb-page-sub">Insights and visualizations from your chat data</div>', unsafe_allow_html=True)

    user = st.session_state.user
    coll_id = st.session_state.active_collection

    if not coll_id:
        st.markdown('<div class="rb-alert rb-alert-info">Select a collection to view analytics.</div>', unsafe_allow_html=True)
        return

    if not PANDAS_OK or not PLOTLY_OK:
        st.markdown('<div class="rb-alert rb-alert-warn">⚠️ Analytics requires pandas and plotly. Check requirements.txt.</div>', unsafe_allow_html=True)
        return


    conn = get_db()
    df = pd.DataFrame(conn.execute(
        "SELECT sender, timestamp, content FROM messages_raw WHERE collection_id=?", (coll_id,)
    ).fetchall(), columns=["sender", "timestamp", "content"])
    conn.close()

    if df.empty:
        st.info("No data found.")
        return

    # Metrics row
    total_msgs = len(df)
    unique_senders = df["sender"].nunique()
    avg_len = int(df["content"].str.len().mean())
    conn2 = get_db()
    coll_info = conn2.execute("SELECT name FROM chat_collections WHERE id=?", (coll_id,)).fetchone()
    conn2.close()
    coll_name_val = coll_info[0] if coll_info else "N/A"

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, total_msgs, "Total Messages"),
        (m2, unique_senders, "Unique Senders"),
        (m3, avg_len, "Avg Msg Length"),
        (m4, coll_name_val[:12], "Collection"),
    ]:
        with col:
            st.markdown(f'<div class="rb-metric"><div class="rb-metric-val">{val}</div><div class="rb-metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["👥 By Sender", "📅 Timeline", "☁️ Keywords", "🔥 Heatmap"])

    theme = st.session_state.get("theme", "dark")
    plot_bg = "#161a24" if theme == "dark" else "#ffffff"
    plot_color = "#e8eaf0" if theme == "dark" else "#1a1f2e"
    palette = ["#6c63ff", "#00d4aa", "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]

    with tab1:
        top_senders = df["sender"].value_counts().head(10).reset_index()
        top_senders.columns = ["Sender", "Messages"]
        fig = px.bar(top_senders, x="Messages", y="Sender", orientation="h",
                     color="Messages", color_continuous_scale=["#6c63ff", "#00d4aa"],
                     title="Top 10 Most Active Senders")
        fig.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=plot_color,
                          yaxis_categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["timestamp"], errors="coerce").dt.date
        daily = df2.groupby("date").size().reset_index(name="count")
        fig2 = px.line(daily, x="date", y="count", title="Daily Message Volume",
                       color_discrete_sequence=["#6c63ff"])
        fig2.update_traces(fill="tozeroy", fillcolor="rgba(108,99,255,0.1)")
        fig2.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=plot_color)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        from collections import Counter
        import re as re2
        all_words = " ".join(df["content"].fillna("")).lower()
        words = re2.findall(r'\b[a-z]{4,}\b', all_words)
        stop = {"that","this","with","from","have","will","been","were","they","your","what","when","just","about","some","more","also","then","into","their","which","would","there"}
        word_freq = Counter(w for w in words if w not in stop)
        top_words = pd.DataFrame(word_freq.most_common(20), columns=["word", "count"])
        fig3 = px.bar(top_words, x="word", y="count", title="Top Keywords",
                      color="count", color_continuous_scale=["#6c63ff", "#00d4aa"])
        fig3.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=plot_color)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        df3 = df.copy()
        df3["dt"] = pd.to_datetime(df3["timestamp"], errors="coerce")
        df3 = df3.dropna(subset=["dt"])
        if not df3.empty:
            df3["hour"] = df3["dt"].dt.hour
            df3["dow"] = df3["dt"].dt.day_name()
            heatmap_data = df3.groupby(["dow", "hour"]).size().reset_index(name="count")
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heatmap_pivot = heatmap_data.pivot(index="dow", columns="hour", values="count").reindex(day_order).fillna(0)
            fig4 = px.imshow(heatmap_pivot, title="Activity Heatmap (Day × Hour)",
                             color_continuous_scale=["#1e2333", "#6c63ff", "#00d4aa"],
                             aspect="auto")
            fig4.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=plot_color)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Timestamps not parseable for heatmap.")

# ─── MEMORY PAGE ──────────────────────────────────────────────────────────────
def page_memory():
    st.markdown('<div class="rb-page-title">Memory</div>', unsafe_allow_html=True)
    st.markdown('<div class="rb-page-sub">Saved queries, bookmarks, and AI-generated summaries</div>', unsafe_allow_html=True)

    user = st.session_state.user
    tab1, tab2 = st.tabs(["📌 Saved Q&A", "🤖 AI Summary"])

    with tab1:
        queries = get_saved_queries(user["id"])
        if not queries:
            st.markdown('<div class="rb-alert rb-alert-info">No saved queries yet. Ask a question and click "Save".</div>', unsafe_allow_html=True)
        for q in queries:
            with st.expander(f"🔖 {q['query'][:60]}… · {q['saved_at'][:10]}"):
                st.markdown(q["answer"])
                if st.button("🗑️ Delete", key=f"del_sq_{q['id']}"):
                    conn = get_db()
                    conn.execute("DELETE FROM saved_queries WHERE id=?", (q["id"],))
                    conn.commit()
                    conn.close()
                    st.rerun()

    with tab2:
        coll_id = st.session_state.active_collection
        if not coll_id:
            st.markdown('<div class="rb-alert rb-alert-info">Select a collection first.</div>', unsafe_allow_html=True)
            return
        if st.button("✨ Generate Summary", type="primary"):
            msgs = get_messages(coll_id, limit=100)
            if not msgs:
                st.error("No messages found.")
                return
            sample_text = "\n".join([f"[{m['timestamp']}] {m['sender']}: {m['content']}" for m in msgs[:80]])
            client = get_openai_client()
            if not client:
                st.error("OpenAI key missing.")
                return
            with st.spinner("Generating summary…"):
                try:
                    resp = client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a chat summarizer. Produce a concise, well-structured markdown summary of the conversation including: main topics discussed, key decisions, action items, and notable participants."},
                            {"role": "user", "content": f"Summarize this chat:\n\n{sample_text}"}
                        ],
                        max_tokens=600,
                    )
                    summary = resp.choices[0].message.content
                    st.markdown(summary)
                    if st.button("📌 Save Summary"):
                        save_query(user["id"], "AI Summary", summary)
                        st.success("Saved!")
                except Exception as e:
                    st.error(f"Error: {e}")

# ─── SETTINGS PAGE ────────────────────────────────────────────────────────────
def page_settings():
    st.markdown('<div class="rb-page-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="rb-page-sub">Configure RecallBot to your preferences</div>', unsafe_allow_html=True)

    user = st.session_state.user

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        st.markdown("#### 🔑 OpenAI API Key")
        current_key = st.session_state.get("openai_key", "")
        masked = ("*" * (len(current_key) - 4) + current_key[-4:]) if len(current_key) > 8 else current_key
        st.markdown(f'<div class="rb-badge rb-badge-{"green" if current_key else "red"}">{("✅ Set: " + masked) if current_key else "❌ Not set"}</div><br>', unsafe_allow_html=True)
        new_key = st.text_input("Enter API key", type="password", placeholder="sk-...")
        if st.button("💾 Save Key", type="primary"):
            st.session_state.openai_key = new_key
            os.environ["OPENAI_API_KEY"] = new_key
            st.success("Key saved for this session!")
        st.markdown("""<div style="font-size:0.78rem;color:#8892a4;margin-top:8px;">
        For persistent storage, set <code>OPENAI_API_KEY</code> in your <code>.env</code> file or Streamlit secrets.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        st.markdown("#### 🎨 Theme")
        curr_theme = st.session_state.get("theme", "dark")
        new_theme = st.radio("Choose theme", ["dark", "light"], index=0 if curr_theme == "dark" else 1, horizontal=True)
        if new_theme != curr_theme:
            st.session_state.theme = new_theme
            conn = get_db()
            conn.execute("UPDATE users SET theme=? WHERE id=?", (new_theme, user["id"]))
            conn.commit()
            conn.close()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        st.markdown("#### 🧹 Data Management")
        colls = get_user_collections(user["id"])
        if colls:
            del_coll = st.selectbox("Select collection to delete", [c["name"] for c in colls], key="del_coll_select")
            if st.button("🗑️ Delete Collection", type="primary"):
                coll_obj = next(c for c in colls if c["name"] == del_coll)
                conn = get_db()
                conn.execute("DELETE FROM messages_raw WHERE collection_id=?", (coll_obj["id"],))
                conn.execute("DELETE FROM chat_collections WHERE id=?", (coll_obj["id"],))
                conn.commit()
                conn.close()
                # Remove from chroma
                client = get_chroma_client()
                if client:
                    try:
                        client.delete_collection(collection_name(user["id"], coll_obj["id"]))
                    except Exception:
                        pass
                if st.session_state.active_collection == coll_obj["id"]:
                    st.session_state.active_collection = None
                st.success(f"Deleted '{del_coll}'")
                st.rerun()
        else:
            st.markdown('<div style="color:#8892a4;font-size:0.88rem;">No collections to manage.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        st.markdown("#### ℹ️ System Info")
        st.markdown(f"""
| Component | Status |
|-----------|--------|
| OpenAI | {"✅" if OPENAI_OK else "❌ pip install openai"} |
| ChromaDB | {"✅" if CHROMA_OK else "❌ pip install chromadb"} |
| SentenceTransformers | {"✅" if ST_OK else "❌ pip install sentence-transformers"} |
| LangChain | {"✅" if LC_OK else "❌ pip install langchain"} |
""")
        st.markdown('</div>', unsafe_allow_html=True)

# ─── DEMO USER ────────────────────────────────────────────────────────────────
def ensure_demo_user():
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username='demo'").fetchone()
    if not existing:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, email, password_hash) VALUES (?,?,?)",
            ("demo", "demo@recallbot.ai", hash_password("demo123"))
        )
        conn.commit()
    conn.close()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    init_db()
    ensure_demo_user()
    init_session()
    inject_css()

    if not st.session_state.user:
        page_auth()
        return

    render_sidebar()

    page = st.session_state.page
    if page == "chat":
        page_chat()
    elif page == "upload":
        page_upload()
    elif page == "search":
        page_search()
    elif page == "analytics":
        page_analytics()
    elif page == "memory":
        page_memory()
    elif page == "settings":
        page_settings()

if __name__ == "__main__":
    main()
