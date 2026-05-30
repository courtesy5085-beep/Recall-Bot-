"""
RecallBot Ultra — AI-Powered Memory Assistant for Team Chats
============================================================
Production-grade Streamlit SaaS application.
"""

import streamlit as st
import os, re, json, zipfile, sqlite3, hashlib, secrets, time, datetime, io
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dotenv import load_dotenv
load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RecallBot — AI Memory Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Environment / secrets ─────────────────────────────────────────────────────
def get_openai_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")

# ── Data directory ─────────────────────────────────────────────────────────────
DATA_DIR = Path("recallbot_data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH  = DATA_DIR / "recallbot.db"
CHROMA_PATH = DATA_DIR / "chroma_db"

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Futuristic Glassmorphism UI
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
  --bg-primary:    #07070E;
  --bg-secondary:  #0F0F1A;
  --bg-card:       rgba(255,255,255,0.04);
  --bg-glass:      rgba(255,255,255,0.06);
  --border:        rgba(0,245,255,0.15);
  --border-bright: rgba(0,245,255,0.4);
  --neon-cyan:     #00F5FF;
  --neon-purple:   #B44FFF;
  --neon-green:    #00FF9F;
  --neon-pink:     #FF2D78;
  --text-primary:  #EEF0FF;
  --text-muted:    rgba(238,240,255,0.5);
  --text-dim:      rgba(238,240,255,0.3);
  --glow-cyan:     0 0 20px rgba(0,245,255,0.3);
  --glow-purple:   0 0 20px rgba(180,79,255,0.3);
  --radius-sm:     8px;
  --radius-md:     14px;
  --radius-lg:     20px;
  --radius-xl:     28px;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-primary) !important;
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text-primary);
}

/* Animated background grid */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  pointer-events: none;
  z-index: 0;
  animation: gridPulse 8s ease-in-out infinite;
}

@keyframes gridPulse {
  0%, 100% { opacity: 0.4; }
  50%       { opacity: 1;   }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A0A15 0%, #080810 100%) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px);
}

[data-testid="stSidebar"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-purple), transparent);
  animation: scanline 3s linear infinite;
}

@keyframes scanline {
  0%   { opacity: 0.3; }
  50%  { opacity: 1;   }
  100% { opacity: 0.3; }
}

/* ── Typography ── */
h1, h2, h3 {
  font-family: 'Syne', sans-serif !important;
  letter-spacing: -0.02em;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(180,79,255,0.12)) !important;
  border: 1px solid var(--border-bright) !important;
  color: var(--neon-cyan) !important;
  border-radius: var(--radius-md) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.25s ease !important;
  position: relative;
  overflow: hidden;
}

.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,245,255,0.25), rgba(180,79,255,0.25)) !important;
  border-color: var(--neon-cyan) !important;
  box-shadow: var(--glow-cyan) !important;
  transform: translateY(-2px) !important;
}

.stButton > button:active {
  transform: translateY(0px) !important;
}

/* Primary button variant */
.btn-primary > button {
  background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple)) !important;
  color: #000 !important;
  border: none !important;
  font-weight: 700 !important;
}

/* ── Text Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--neon-cyan) !important;
  box-shadow: 0 0 0 3px rgba(0,245,255,0.1) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 4px;
}

.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 500 !important;
  padding: 0.6rem 1.2rem !important;
  transition: all 0.2s !important;
}

.stTabs [aria-selected="true"] {
  background: rgba(0,245,255,0.08) !important;
  color: var(--neon-cyan) !important;
  border-bottom: 2px solid var(--neon-cyan) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--bg-card) !important;
  transition: border-color 0.2s, background 0.2s;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--neon-cyan) !important;
  background: rgba(0,245,255,0.04) !important;
}

/* ── Select / Multiselect ── */
[data-baseweb="select"] {
  background: var(--bg-card) !important;
}

[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.04) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-md) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 1rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(0,245,255,0.2);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(0,245,255,0.4); }

/* ── Custom component classes ── */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  backdrop-filter: blur(20px);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.glass-card:hover {
  border-color: rgba(0,245,255,0.35);
  box-shadow: 0 8px 32px rgba(0,245,255,0.08);
}

.neon-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: rgba(0,245,255,0.1);
  border: 1px solid rgba(0,245,255,0.3);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--neon-cyan);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.neon-badge.purple {
  background: rgba(180,79,255,0.1);
  border-color: rgba(180,79,255,0.3);
  color: var(--neon-purple);
}

.neon-badge.green {
  background: rgba(0,255,159,0.1);
  border-color: rgba(0,255,159,0.3);
  color: var(--neon-green);
}

.message-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-left: 3px solid var(--neon-cyan);
  border-radius: var(--radius-md);
  padding: 0.9rem 1.2rem;
  margin: 0.5rem 0;
  transition: all 0.2s;
}

.message-card:hover {
  background: rgba(0,245,255,0.04);
  border-left-color: var(--neon-purple);
}

.message-meta {
  font-size: 0.75rem;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 4px;
}

.message-text {
  color: var(--text-primary);
  font-size: 0.92rem;
  line-height: 1.5;
}

.ai-answer-box {
  background: linear-gradient(135deg, rgba(0,245,255,0.05), rgba(180,79,255,0.05));
  border: 1px solid var(--border-bright);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
}

.ai-answer-box::before {
  content: '🧠 AI';
  position: absolute;
  top: 0.8rem; right: 1rem;
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--neon-cyan);
  letter-spacing: 0.1em;
  opacity: 0.7;
}

.logo-text {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.03em;
}

.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 0.2rem;
}

.section-subtitle {
  font-size: 0.88rem;
  color: var(--text-muted);
  margin-bottom: 1.5rem;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.2rem;
  text-align: center;
  transition: all 0.25s;
}

.stat-card:hover {
  border-color: var(--neon-cyan);
  box-shadow: var(--glow-cyan);
  transform: translateY(-3px);
}

.stat-value {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.stat-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 4px;
}

.bookmark-card {
  background: rgba(180,79,255,0.06);
  border: 1px solid rgba(180,79,255,0.2);
  border-radius: var(--radius-md);
  padding: 1rem 1.2rem;
  margin: 0.5rem 0;
}

.typing-cursor::after {
  content: '▋';
  display: inline-block;
  animation: blink 1s step-end infinite;
  color: var(--neon-cyan);
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0; }
}

.status-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--neon-green);
  box-shadow: 0 0 8px var(--neon-green);
  animation: pulse 2s ease-in-out infinite;
  margin-right: 6px;
}

@keyframes pulse {
  0%, 100% { transform: scale(1);   opacity: 1; }
  50%       { transform: scale(1.3); opacity: 0.7; }
}

.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-bright), transparent);
  margin: 1.5rem 0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Progress bar */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple)) !important;
  border-radius: 4px !important;
}

/* Slider */
.stSlider > div > div > div > div {
  background: var(--neon-cyan) !important;
}

/* Info/warning/error boxes */
.stAlert {
  border-radius: var(--radius-md) !important;
  border: none !important;
}

/* Spinner */
.stSpinner > div {
  border-top-color: var(--neon-cyan) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border-radius: var(--radius-md) !important;
  overflow: hidden;
}

/* Login specific */
.auth-container {
  max-width: 420px;
  margin: 0 auto;
  padding: 2.5rem;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  backdrop-filter: blur(20px);
}

.auth-logo {
  text-align: center;
  margin-bottom: 2rem;
}

.strength-bar {
  height: 4px;
  border-radius: 2px;
  transition: all 0.3s;
  margin-top: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE — SQLite Auth + Bookmarks + Saved Answers
# ═══════════════════════════════════════════════════════════════════════════════
def init_db():
    """Initialize SQLite database with all required tables."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            email    TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            salt     TEXT    NOT NULL,
            created  TEXT    NOT NULL,
            last_login TEXT
        )
    """)

    # Sessions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token    TEXT PRIMARY KEY,
            user_id  INTEGER NOT NULL,
            username TEXT    NOT NULL,
            created  TEXT    NOT NULL,
            expires  TEXT    NOT NULL
        )
    """)

    # Bookmarks table
    c.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            content  TEXT    NOT NULL,
            sender   TEXT,
            timestamp TEXT,
            source   TEXT,
            tag      TEXT,
            created  TEXT    NOT NULL
        )
    """)

    # Saved answers table
    c.execute("""
        CREATE TABLE IF NOT EXISTS saved_answers (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            question TEXT    NOT NULL,
            answer   TEXT    NOT NULL,
            created  TEXT    NOT NULL
        )
    """)

    # Indexed chats table (metadata)
    c.execute("""
        CREATE TABLE IF NOT EXISTS indexed_chats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            filename    TEXT    NOT NULL,
            chat_type   TEXT    NOT NULL,
            msg_count   INTEGER,
            indexed_at  TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ── Auth helpers ───────────────────────────────────────────────────────────────
def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations=260_000
    ).hex()

def validate_email(email: str) -> bool:
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email))

def password_strength(pw: str) -> tuple[int, str]:
    """Returns (score 0-4, label)."""
    score = 0
    if len(pw) >= 8:  score += 1
    if re.search(r'[A-Z]', pw): score += 1
    if re.search(r'[0-9]', pw): score += 1
    if re.search(r'[^A-Za-z0-9]', pw): score += 1
    labels = ["Very Weak", "Weak", "Fair", "Strong", "Very Strong"]
    return score, labels[score]

def signup_user(username: str, email: str, password: str) -> tuple[bool, str]:
    if not validate_email(email):
        return False, "Invalid email address."
    score, _ = password_strength(password)
    if score < 2:
        return False, "Password too weak. Use 8+ chars with uppercase and numbers."

    salt = secrets.token_hex(32)
    pw_hash = hash_password(password, salt)
    now = datetime.datetime.utcnow().isoformat()

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("INSERT INTO users (username,email,password,salt,created) VALUES (?,?,?,?,?)",
                     (username.strip(), email.strip().lower(), pw_hash, salt, now))
        conn.commit()
        conn.close()
        return True, "Account created!"
    except sqlite3.IntegrityError as e:
        return False, "Username or email already exists."

def login_user(username: str, password: str) -> tuple[bool, str, dict]:
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("SELECT id,username,email,password,salt FROM users WHERE username=?",
                       (username.strip(),)).fetchone()
    if not row:
        conn.close()
        return False, "User not found.", {}
    uid, uname, email, pw_hash, salt = row
    if hash_password(password, salt) != pw_hash:
        conn.close()
        return False, "Incorrect password.", {}

    # Create session token
    token = secrets.token_hex(48)
    now   = datetime.datetime.utcnow()
    exp   = (now + datetime.timedelta(days=7)).isoformat()

    conn.execute("INSERT INTO sessions (token,user_id,username,created,expires) VALUES (?,?,?,?,?)",
                 (token, uid, uname, now.isoformat(), exp))
    conn.execute("UPDATE users SET last_login=? WHERE id=?", (now.isoformat(), uid))
    conn.commit()
    conn.close()

    return True, "Login successful!", {"user_id": uid, "username": uname, "email": email, "token": token}

def validate_session(token: str) -> dict | None:
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("SELECT user_id,username,expires FROM sessions WHERE token=?", (token,)).fetchone()
    conn.close()
    if not row:
        return None
    uid, uname, exp = row
    if datetime.datetime.utcnow() > datetime.datetime.fromisoformat(exp):
        return None
    return {"user_id": uid, "username": uname, "token": token}

def logout_user(token: str):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()

# ── Bookmark helpers ───────────────────────────────────────────────────────────
def save_bookmark(user_id, content, sender="", timestamp="", source="", tag=""):
    now = datetime.datetime.utcnow().isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO bookmarks (user_id,content,sender,timestamp,source,tag,created) VALUES (?,?,?,?,?,?,?)",
                 (user_id, content, sender, timestamp, source, tag, now))
    conn.commit()
    conn.close()

def get_bookmarks(user_id):
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT id,content,sender,timestamp,source,tag,created FROM bookmarks WHERE user_id=? ORDER BY created DESC",
                        (user_id,)).fetchall()
    conn.close()
    return rows

def delete_bookmark(bookmark_id, user_id):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM bookmarks WHERE id=? AND user_id=?", (bookmark_id, user_id))
    conn.commit()
    conn.close()

def save_answer(user_id, question, answer):
    now = datetime.datetime.utcnow().isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO saved_answers (user_id,question,answer,created) VALUES (?,?,?,?)",
                 (user_id, question, answer, now))
    conn.commit()
    conn.close()

def get_saved_answers(user_id):
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT id,question,answer,created FROM saved_answers WHERE user_id=? ORDER BY created DESC",
                        (user_id,)).fetchall()
    conn.close()
    return rows

def save_chat_meta(user_id, filename, chat_type, msg_count):
    now = datetime.datetime.utcnow().isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO indexed_chats (user_id,filename,chat_type,msg_count,indexed_at) VALUES (?,?,?,?,?)",
                 (user_id, filename, chat_type, msg_count, now))
    conn.commit()
    conn.close()

def get_indexed_chats(user_id):
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT filename,chat_type,msg_count,indexed_at FROM indexed_chats WHERE user_id=? ORDER BY indexed_at DESC",
                        (user_id,)).fetchall()
    conn.close()
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_whatsapp(text: str) -> list[dict]:
    """Parse WhatsApp .txt export into list of message dicts."""
    messages = []
    # Patterns: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD
    patterns = [
        r'^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s*[-–]\s*([^:]+):\s*(.+)$',
        r'^\[(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s*([^:]+):\s*(.+)$',
    ]
    compiled = [re.compile(p, re.MULTILINE) for p in patterns]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for pat in compiled:
            m = pat.match(line)
            if m:
                date_s, time_s, sender, msg = m.groups()
                # Skip system messages & media
                if '<Media omitted>' in msg or msg.startswith('~'):
                    msg = '[Media]'
                messages.append({
                    "timestamp": f"{date_s} {time_s}",
                    "sender": sender.strip(),
                    "text": msg.strip(),
                    "source": "whatsapp",
                    "channel": "chat",
                })
                break
    return messages

def parse_slack_json(data: dict | list) -> list[dict]:
    """Parse Slack JSON export (channel messages list or full export dict)."""
    messages = []

    def extract_msgs(msgs, channel="general"):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if m.get("type") != "message":
                continue
            if m.get("subtype") in ("channel_join", "channel_leave", "bot_add"):
                continue
            ts_raw = m.get("ts", "")
            try:
                ts = datetime.datetime.fromtimestamp(float(ts_raw)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = ts_raw
            text = m.get("text", "").strip()
            # Clean Slack formatting
            text = re.sub(r'<@\w+>', '@user', text)
            text = re.sub(r'<#\w+\|([^>]+)>', r'#\1', text)
            text = re.sub(r'<https?://[^|>]+\|([^>]+)>', r'\1', text)
            text = re.sub(r'<https?://[^>]+>', '[link]', text)
            if text:
                messages.append({
                    "timestamp": ts,
                    "sender": m.get("user", m.get("username", "Unknown")),
                    "text": text,
                    "source": "slack",
                    "channel": channel,
                })

    if isinstance(data, list):
        extract_msgs(data, "general")
    elif isinstance(data, dict):
        for channel, msgs in data.items():
            if isinstance(msgs, list):
                extract_msgs(msgs, channel)

    return messages

def parse_slack_zip(zip_bytes: bytes) -> list[dict]:
    """Parse Slack zip export (multiple JSON files per channel)."""
    messages = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.endswith(".json"):
                parts = name.replace("\\", "/").split("/")
                channel = parts[-2] if len(parts) >= 2 else "general"
                try:
                    with z.open(name) as f:
                        data = json.load(f)
                    messages.extend(parse_slack_json(data if isinstance(data, list) else {channel: data}))
                except Exception:
                    pass
    return messages

# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — ChromaDB + SentenceTransformers
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client

def get_collection(user_id: int):
    """Get or create a per-user ChromaDB collection."""
    client = get_chroma_client()
    name = f"user_{user_id}"
    try:
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    except Exception:
        return client.get_collection(name=name)

def index_messages(user_id: int, messages: list[dict], progress_callback=None) -> int:
    """Chunk and embed messages into ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    model      = get_embedding_model()
    collection = get_collection(user_id)

    # Get existing IDs to avoid duplicates
    try:
        existing = set(collection.get()["ids"])
    except Exception:
        existing = set()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)

    docs, ids, metas = [], [], []
    total = len(messages)

    for i, msg in enumerate(messages):
        if progress_callback:
            progress_callback(i / total)

        text = msg.get("text", "").strip()
        if len(text) < 5:
            continue

        chunks = splitter.split_text(text)
        for j, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{user_id}_{msg['timestamp']}_{msg['sender']}_{j}".encode()).hexdigest()
            if doc_id in existing:
                continue
            docs.append(chunk)
            ids.append(doc_id)
            metas.append({
                "sender":    msg.get("sender", "Unknown"),
                "timestamp": msg.get("timestamp", ""),
                "source":    msg.get("source", "unknown"),
                "channel":   msg.get("channel", ""),
                "original":  text[:500],
            })

    if not docs:
        return 0

    # Batch embed
    BATCH = 64
    embeddings = []
    for start in range(0, len(docs), BATCH):
        batch_emb = model.encode(docs[start:start+BATCH], show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())

    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    return len(docs)

def semantic_search(user_id: int, query: str, k: int = 8, filters: dict = None) -> list[dict]:
    """Hybrid search: semantic + BM25 re-ranked."""
    from rank_bm25 import BM25Okapi

    model      = get_embedding_model()
    collection = get_collection(user_id)

    try:
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs  = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])
        all_ids   = all_data.get("ids", [])
    except Exception:
        return []

    if not all_docs:
        return []

    # Apply metadata filters
    if filters:
        filtered = []
        for doc, meta, did in zip(all_docs, all_metas, all_ids):
            match = True
            if filters.get("sender") and meta.get("sender") != filters["sender"]:
                match = False
            if filters.get("source") and meta.get("source") != filters["source"]:
                match = False
            if match:
                filtered.append((doc, meta, did))
        all_docs  = [x[0] for x in filtered]
        all_metas = [x[1] for x in filtered]
        all_ids   = [x[2] for x in filtered]

    if not all_docs:
        return []

    # BM25 keyword search
    tokenized = [d.lower().split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Semantic search
    q_emb = model.encode([query], show_progress_bar=False)[0].tolist()
    try:
        sem_results = collection.query(
            query_embeddings=[q_emb],
            n_results=min(k * 2, len(all_docs)),
            include=["documents", "metadatas", "distances"],
        )
        sem_ids  = sem_results["ids"][0]
        sem_dist = sem_results["distances"][0]
        # Normalise semantic scores
        sem_score_map = {}
        for sid, dist in zip(sem_ids, sem_dist):
            sem_score_map[sid] = 1 - dist  # cosine similarity
    except Exception:
        sem_score_map = {}

    # Combine scores
    combined = []
    for i, (doc, meta, did) in enumerate(zip(all_docs, all_metas, all_ids)):
        bm25_norm = float(bm25_scores[i]) / (max(bm25_scores) + 1e-9)
        sem_norm  = sem_score_map.get(did, 0.0)
        score     = 0.4 * bm25_norm + 0.6 * sem_norm
        combined.append({"id": did, "text": doc, "meta": meta, "score": score})

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:k]

def get_all_messages_df(user_id: int) -> pd.DataFrame:
    """Return all indexed messages as a DataFrame for analytics."""
    collection = get_collection(user_id)
    try:
        data = collection.get(include=["documents", "metadatas"])
        rows = []
        for doc, meta in zip(data["documents"], data["metadatas"]):
            rows.append({
                "text": doc,
                "sender": meta.get("sender", "Unknown"),
                "timestamp": meta.get("timestamp", ""),
                "source": meta.get("source", ""),
                "channel": meta.get("channel", ""),
            })
        df = pd.DataFrame(rows)
        if not df.empty and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# AI ENGINE — GPT-4o-mini via OpenAI
# ═══════════════════════════════════════════════════════════════════════════════
def build_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        m = r["meta"]
        lines.append(f"[{i}] [{m.get('timestamp','')}] {m.get('sender','?')}: {r['text']}")
    return "\n".join(lines)

def ask_ai_streaming(question: str, context: str, api_key: str):
    """Yield streaming GPT-4o-mini tokens."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    system = (
        "You are RecallBot, an expert AI memory assistant for team communications. "
        "You have been given relevant chat messages as context. "
        "Answer the user's question concisely, citing message numbers like [1] [2] when referencing specific messages. "
        "If context doesn't contain the answer, say so honestly. "
        "Format your answer in clean Markdown."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Context messages:\n{context}\n\nQuestion: {question}"},
    ]
    with client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        max_tokens=800,
        temperature=0.3,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

def generate_summary(messages: list[dict], api_key: str) -> str:
    """Generate an AI summary of a batch of messages."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    sample = messages[:80]
    text = "\n".join(f"{m['sender']}: {m['text']}" for m in sample)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at summarising team conversations. Be concise and insightful."},
            {"role": "user",   "content": f"Summarise these chat messages in 4-6 bullet points:\n\n{text}"},
        ],
        max_tokens=400,
        temperature=0.4,
    )
    return resp.choices[0].message.content

def extract_action_items(messages: list[dict], api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    sample = messages[:80]
    text = "\n".join(f"{m['sender']}: {m['text']}" for m in sample)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract action items, tasks, and deadlines from these messages. Format as a numbered list with owner and due date if mentioned."},
            {"role": "user", "content": text},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return resp.choices[0].message.content

def suggest_followups(question: str, answer: str, api_key: str) -> list[str]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate 3 short follow-up questions (max 10 words each) the user might want to ask next. Return as JSON array of strings only."},
            {"role": "user", "content": f"Q: {question}\nA: {answer}"},
        ],
        max_tokens=100,
        temperature=0.7,
    )
    try:
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def make_plotly_dark():
    return dict(
        plot_bgcolor  ="rgba(0,0,0,0)",
        paper_bgcolor ="rgba(0,0,0,0)",
        font_color    ="#EEF0FF",
        font_family   ="Space Grotesk",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
    )

def build_analytics(df: pd.DataFrame):
    """Build Plotly figures from messages DataFrame."""
    figs = {}

    if df.empty:
        return figs

    # 1) Messages per user (bar)
    if "sender" in df.columns:
        top = df["sender"].value_counts().head(15).reset_index()
        top.columns = ["User", "Messages"]
        fig_users = px.bar(
            top, x="Messages", y="User", orientation="h",
            color="Messages",
            color_continuous_scale=["#B44FFF","#00F5FF"],
            title="Most Active Users",
        )
        fig_users.update_layout(**make_plotly_dark(), coloraxis_showscale=False)
        figs["users"] = fig_users

    # 2) Activity over time (line)
    if "datetime" in df.columns:
        ts_df = df.dropna(subset=["datetime"])
        if not ts_df.empty:
            ts_df = ts_df.copy()
            ts_df["date"] = ts_df["datetime"].dt.date
            daily = ts_df.groupby("date").size().reset_index(name="count")
            fig_line = px.area(
                daily, x="date", y="count",
                title="Message Volume Over Time",
                color_discrete_sequence=["#00F5FF"],
            )
            fig_line.update_traces(fill="tozeroy", fillcolor="rgba(0,245,255,0.08)")
            fig_line.update_layout(**make_plotly_dark())
            figs["timeline"] = fig_line

    # 3) Activity heatmap (day-of-week × hour)
    if "datetime" in df.columns:
        ts_df = df.dropna(subset=["datetime"]).copy()
        if not ts_df.empty:
            ts_df["hour"] = ts_df["datetime"].dt.hour
            ts_df["dow"]  = ts_df["datetime"].dt.day_name()
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heat = ts_df.groupby(["dow","hour"]).size().reset_index(name="count")
            pivot = heat.pivot(index="dow", columns="hour", values="count").reindex(order).fillna(0)
            fig_heat = px.imshow(
                pivot,
                color_continuous_scale=[[0,"#0A0A1A"],[0.5,"#B44FFF"],[1,"#00F5FF"]],
                title="Activity Heatmap (Day × Hour)",
                aspect="auto",
            )
            fig_heat.update_layout(**make_plotly_dark())
            figs["heatmap"] = fig_heat

    # 4) Source breakdown (donut)
    if "source" in df.columns:
        src = df["source"].value_counts().reset_index()
        src.columns = ["Source","Count"]
        fig_pie = px.pie(
            src, names="Source", values="Count",
            hole=0.6,
            color_discrete_sequence=["#00F5FF","#B44FFF","#00FF9F","#FF2D78"],
            title="Chat Sources",
        )
        fig_pie.update_layout(**make_plotly_dark())
        figs["sources"] = fig_pie

    return figs

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "auth":            None,     # dict or None
        "chat_messages":   [],       # indexed messages list
        "conversation":    [],       # Q&A history
        "last_answer":     "",
        "last_question":   "",
        "last_sources":    [],
        "followups":       [],
        "page":            "ask",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Restore session from cookie-like token in session_state ───────────────────
if "session_token" in st.session_state and not st.session_state.auth:
    info = validate_session(st.session_state.session_token)
    if info:
        st.session_state.auth = info

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH SCREENS
# ═══════════════════════════════════════════════════════════════════════════════
def render_auth():
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 1rem 0;">
        <div class="logo-text" style="font-size:2.5rem;">🧠 RecallBot</div>
        <p style="color:var(--text-muted); margin-top:0.5rem; font-size:1rem;">
            AI-Powered Memory for Team Chats
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        tab1, tab2 = st.tabs(["  Sign In  ", "  Create Account  "])

        with tab1:  # LOGIN
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="li_user", placeholder="your_username")
            password = st.text_input("Password", type="password", key="li_pass", placeholder="••••••••")
            remember = st.checkbox("Remember me for 7 days", value=True, key="li_remember")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", key="btn_login", use_container_width=True):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    ok, msg, info = login_user(username, password)
                    if ok:
                        st.session_state.auth = info
                        if remember:
                            st.session_state.session_token = info["token"]
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("""
            <div style="text-align:center; margin-top:1rem; font-size:0.8rem; color:var(--text-dim);">
                Demo: Sign up for a new account to get started.
            </div>""", unsafe_allow_html=True)

        with tab2:  # SIGNUP
            st.markdown("<br>", unsafe_allow_html=True)
            su_user  = st.text_input("Username", key="su_user", placeholder="cooluser42")
            su_email = st.text_input("Email",    key="su_email", placeholder="you@example.com")
            su_pass  = st.text_input("Password", type="password", key="su_pass", placeholder="Min 8 chars")

            if su_pass:
                score, label = password_strength(su_pass)
                colors = ["#FF2D78","#FF8C00","#FFD700","#00FF9F","#00F5FF"]
                widths = [20, 40, 60, 80, 100]
                st.markdown(f"""
                <div style="font-size:0.75rem; color:var(--text-muted); margin-top:-8px;">
                    Strength: <strong style="color:{colors[score]}">{label}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.08); border-radius:2px; height:4px; margin-top:4px;">
                    <div style="width:{widths[score]}%; background:{colors[score]}; height:100%; border-radius:2px; transition:width 0.3s;"></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", key="btn_signup", use_container_width=True):
                if not su_user or not su_email or not su_pass:
                    st.error("Please fill in all fields.")
                else:
                    ok, msg = signup_user(su_user, su_email, su_pass)
                    if ok:
                        st.success(f"✅ {msg} You can now sign in.")
                    else:
                        st.error(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP (AUTHENTICATED)
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    auth = st.session_state.auth
    with st.sidebar:
        st.markdown(f'<div class="logo-text" style="font-size:1.4rem; padding: 0.5rem 0 1rem 0;">🧠 RecallBot</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex; align-items:center; padding: 0.6rem; background:rgba(0,245,255,0.05);
             border:1px solid var(--border); border-radius:10px; margin-bottom:1.2rem;">
            <span class="status-dot"></span>
            <div>
                <div style="font-weight:600; font-size:0.9rem;">{auth['username']}</div>
                <div style="font-size:0.72rem; color:var(--text-muted);">Active session</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # API key input
        api_key = get_openai_key()
        if not api_key:
            api_key = st.text_input("🔑 OpenAI API Key", type="password",
                                    key="api_key_input", placeholder="sk-...")
            if api_key:
                st.session_state["runtime_api_key"] = api_key
        else:
            st.session_state["runtime_api_key"] = api_key
            st.markdown('<div class="neon-badge green">✓ API Key Set</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Navigation
        pages = {
            "ask":       ("💬", "Ask RecallBot"),
            "import":    ("📁", "Import Chats"),
            "search":    ("🔍", "Search"),
            "analytics": ("📊", "Analytics"),
            "bookmarks": ("🔖", "Bookmarks"),
            "insights":  ("⚡", "AI Insights"),
        }
        for pid, (icon, label) in pages.items():
            active = "style='background:rgba(0,245,255,0.1); border-color:rgba(0,245,255,0.4); color:var(--neon-cyan);'" if st.session_state.page == pid else ""
            if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True):
                st.session_state.page = pid
                st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Indexed chats summary
        chats = get_indexed_chats(auth["user_id"])
        if chats:
            st.markdown(f'<div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.06em;">Indexed Chats</div>', unsafe_allow_html=True)
            for fname, ctype, mc, idxat in chats[-4:]:
                icon = "📱" if ctype == "whatsapp" else "💼"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:4px 0; font-size:0.78rem;">
                    <span>{icon} {fname[:20]}</span>
                    <span class="neon-badge" style="font-size:0.65rem;">{mc}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↩  Sign Out", key="btn_logout", use_container_width=True):
            logout_user(auth["token"])
            st.session_state.auth = None
            st.session_state.session_token = None
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
def page_import():
    auth    = st.session_state.auth
    api_key = st.session_state.get("runtime_api_key", "")

    st.markdown('<div class="section-title">📁 Import Chats</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload WhatsApp .txt or Slack .json / .zip exports to index them into AI memory.</div>', unsafe_allow_html=True)

    tab_wa, tab_sl = st.tabs(["  📱 WhatsApp  ", "  💼 Slack  "])

    with tab_wa:
        st.markdown("<br>", unsafe_allow_html=True)
        up = st.file_uploader("Upload WhatsApp export (.txt)", type=["txt"], key="wa_up")
        if up:
            st.markdown(f'<div class="neon-badge">📄 {up.name} — {up.size // 1024} KB</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Index into Memory", key="btn_wa_index", use_container_width=True):
                text = up.read().decode("utf-8", errors="ignore")
                with st.spinner("Parsing WhatsApp chat..."):
                    msgs = parse_whatsapp(text)
                if not msgs:
                    st.error("No messages found. Check your file format.")
                    return
                st.success(f"Parsed {len(msgs):,} messages.")
                st.session_state.chat_messages = msgs

                progress = st.progress(0, text="Indexing into AI memory…")
                def upd(v): progress.progress(v, text=f"Indexing… {int(v*100)}%")
                with st.spinner("Embedding…"):
                    n = index_messages(auth["user_id"], msgs, upd)
                progress.progress(1.0, text="Done!")
                save_chat_meta(auth["user_id"], up.name, "whatsapp", len(msgs))
                st.success(f"✅ Indexed {n:,} chunks. RecallBot now remembers this chat!")

                if api_key and len(msgs) >= 5:
                    with st.expander("🤖 AI Summary", expanded=True):
                        with st.spinner("Generating summary…"):
                            summary = generate_summary(msgs, api_key)
                        st.markdown(f'<div class="ai-answer-box">{summary}</div>', unsafe_allow_html=True)

    with tab_sl:
        st.markdown("<br>", unsafe_allow_html=True)
        up2 = st.file_uploader("Upload Slack export (.json or .zip)", type=["json","zip"], key="sl_up")
        if up2:
            st.markdown(f'<div class="neon-badge purple">📄 {up2.name} — {up2.size // 1024} KB</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Index into Memory", key="btn_sl_index", use_container_width=True):
                raw = up2.read()
                with st.spinner("Parsing Slack export…"):
                    if up2.name.endswith(".zip"):
                        msgs = parse_slack_zip(raw)
                    else:
                        data = json.loads(raw.decode("utf-8"))
                        msgs = parse_slack_json(data)

                if not msgs:
                    st.error("No messages found.")
                    return
                st.success(f"Parsed {len(msgs):,} messages.")
                st.session_state.chat_messages = msgs

                progress = st.progress(0, text="Indexing…")
                def upd2(v): progress.progress(v)
                with st.spinner("Embedding…"):
                    n = index_messages(auth["user_id"], msgs, upd2)
                progress.progress(1.0)
                save_chat_meta(auth["user_id"], up2.name, "slack", len(msgs))
                st.success(f"✅ Indexed {n:,} chunks.")

                if api_key and len(msgs) >= 5:
                    with st.expander("🤖 AI Summary", expanded=True):
                        with st.spinner("Generating summary…"):
                            summary = generate_summary(msgs, api_key)
                        st.markdown(f'<div class="ai-answer-box">{summary}</div>', unsafe_allow_html=True)

    # Show existing indexed chats
    chats = get_indexed_chats(auth["user_id"])
    if chats:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Indexed Chats</div>', unsafe_allow_html=True)
        for fname, ctype, mc, idxat in chats:
            icon = "📱" if ctype == "whatsapp" else "💼"
            st.markdown(f"""
            <div class="message-card">
                <div class="message-meta">{icon} {ctype.upper()} · {idxat[:19]}</div>
                <div class="message-text">{fname} — <strong>{mc:,}</strong> messages indexed</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
def page_ask():
    auth    = st.session_state.auth
    api_key = st.session_state.get("runtime_api_key", "")

    st.markdown('<div class="section-title">💬 Ask RecallBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Ask any question about your indexed conversations. Get AI answers with cited sources.</div>', unsafe_allow_html=True)

    if not api_key:
        st.warning("⚠️  Add your OpenAI API key in the sidebar to enable AI answers.")

    # Conversation history
    for item in st.session_state.conversation[-6:]:
        role = item["role"]
        content = item["content"]
        if role == "user":
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin:0.5rem 0;">
                <div style="background:rgba(0,245,255,0.1); border:1px solid rgba(0,245,255,0.25); 
                     border-radius:16px 16px 4px 16px; padding:0.8rem 1.1rem; max-width:75%;
                     color:var(--text-primary); font-size:0.92rem;">
                    {content}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin:0.5rem 0 1rem 0;">
                <div class="ai-answer-box" style="font-size:0.9rem; line-height:1.7;">
                    {content}
                </div>
            </div>""", unsafe_allow_html=True)

    # Follow-up suggestions
    if st.session_state.followups:
        st.markdown('<div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.4rem;">Suggested follow-ups:</div>', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.followups))
        for i, fq in enumerate(st.session_state.followups):
            with cols[i]:
                if st.button(fq, key=f"fq_{i}", use_container_width=True):
                    st.session_state["prefill_q"] = fq
                    st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Query input
    prefill = st.session_state.pop("prefill_q", "") if "prefill_q" in st.session_state else ""
    question = st.text_input(
        "Ask a question…",
        key="ask_input",
        value=prefill,
        placeholder="e.g. What did Alice say about the deployment last week?",
        label_visibility="collapsed",
    )

    col_ask, col_save, col_clear = st.columns([3,1,1])
    with col_ask:
        ask_btn = st.button("Ask RecallBot →", key="btn_ask", use_container_width=True)
    with col_save:
        save_btn = st.button("🔖 Save Answer", key="btn_save_ans", use_container_width=True)
    with col_clear:
        if st.button("🗑 Clear", key="btn_clear_chat", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.followups    = []
            st.rerun()

    if save_btn and st.session_state.last_answer:
        save_answer(auth["user_id"], st.session_state.last_question, st.session_state.last_answer)
        st.success("Answer saved to your bookmarks!")

    if ask_btn and question.strip():
        # Retrieve context
        with st.spinner("Searching memory…"):
            results = semantic_search(auth["user_id"], question, k=6)

        if not results:
            st.warning("No relevant messages found. Import some chats first!")
            return

        context = build_context(results)
        st.session_state.conversation.append({"role": "user", "content": question})

        # Sources expander
        with st.expander(f"📚 {len(results)} Source Messages", expanded=False):
            for i, r in enumerate(results, 1):
                m = r["meta"]
                score_pct = int(r["score"] * 100)
                st.markdown(f"""
                <div class="message-card">
                    <div class="message-meta">
                        [{i}] {m.get('sender','?')} · {m.get('timestamp','')} · 
                        {m.get('source','').upper()} · {m.get('channel','')} 
                        &nbsp; <span class="neon-badge" style="font-size:0.65rem;">{score_pct}% match</span>
                    </div>
                    <div class="message-text">{r['text']}</div>
                </div>""", unsafe_allow_html=True)
                c1, c2 = st.columns([8,1])
                with c2:
                    if st.button("🔖", key=f"bk_{i}_{question[:10]}"):
                        save_bookmark(auth["user_id"], r["text"],
                                      m.get("sender",""), m.get("timestamp",""),
                                      m.get("source",""))
                        st.success("Bookmarked!")

        # AI streaming answer
        if api_key:
            st.markdown('<div class="section-title" style="font-size:1rem; margin-top:1rem;">🧠 AI Answer</div>', unsafe_allow_html=True)
            answer_placeholder = st.empty()
            full_answer = ""

            answer_placeholder.markdown('<div class="ai-answer-box"><span class="typing-cursor"></span></div>', unsafe_allow_html=True)

            for token in ask_ai_streaming(question, context, api_key):
                full_answer += token
                answer_placeholder.markdown(f'<div class="ai-answer-box">{full_answer}</div>', unsafe_allow_html=True)

            st.session_state.conversation.append({"role": "assistant", "content": full_answer})
            st.session_state.last_answer   = full_answer
            st.session_state.last_question = question

            # Follow-ups
            with st.spinner("Generating follow-up suggestions…"):
                fqs = suggest_followups(question, full_answer, api_key)
            st.session_state.followups = fqs
            st.rerun()
        else:
            # No API key — show context only
            st.info("Add your OpenAI API key to get AI-generated answers. Showing raw source messages above.")

# ─────────────────────────────────────────────────────────────────────────────
def page_search():
    auth = st.session_state.auth

    st.markdown('<div class="section-title">🔍 Advanced Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Semantic + keyword hybrid search with filters.</div>', unsafe_allow_html=True)

    with st.form("search_form"):
        query   = st.text_input("Search query", placeholder="meeting, deadline, Bob's update…")
        c1, c2, c3 = st.columns(3)
        with c1:
            top_k = st.slider("Results", 5, 30, 10)
        with c2:
            source_filter = st.selectbox("Source", ["All","whatsapp","slack"])
        with c3:
            sender_filter = st.text_input("Sender filter", placeholder="leave blank for all")
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    if submitted and query.strip():
        filters = {}
        if source_filter != "All":
            filters["source"] = source_filter
        if sender_filter.strip():
            filters["sender"] = sender_filter.strip()

        with st.spinner("Searching…"):
            results = semantic_search(auth["user_id"], query, k=top_k, filters=filters or None)

        if not results:
            st.warning("No results found.")
            return

        st.markdown(f'<div style="margin:0.5rem 0 1rem 0;"><span class="neon-badge green">✓ {len(results)} results</span></div>', unsafe_allow_html=True)

        # Export
        rows = [{"sender": r["meta"].get("sender",""), "timestamp": r["meta"].get("timestamp",""),
                 "source": r["meta"].get("source",""), "text": r["text"], "score": r["score"]} for r in results]
        df_exp = pd.DataFrame(rows)
        csv    = df_exp.to_csv(index=False)
        st.download_button("⬇ Export CSV", csv, "recallbot_results.csv", "text/csv", key="dl_csv")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        for i, r in enumerate(results, 1):
            m    = r["meta"]
            pct  = int(r["score"] * 100)
            # Highlight query terms
            highlighted = r["text"]
            for word in query.split():
                highlighted = re.sub(
                    f"({re.escape(word)})",
                    r'<mark style="background:rgba(0,245,255,0.25); color:#00F5FF; border-radius:3px; padding:0 2px;">\1</mark>',
                    highlighted, flags=re.IGNORECASE
                )

            st.markdown(f"""
            <div class="message-card">
                <div class="message-meta">
                    [{i}] <strong>{m.get('sender','?')}</strong> · {m.get('timestamp','')} · 
                    {m.get('source','').upper()} #{m.get('channel','')}
                    &nbsp; <span class="neon-badge" style="font-size:0.65rem;">{pct}%</span>
                </div>
                <div class="message-text">{highlighted}</div>
            </div>""", unsafe_allow_html=True)

            c_bk, c_ = st.columns([1,6])
            with c_bk:
                if st.button("🔖 Save", key=f"sbk_{i}"):
                    save_bookmark(auth["user_id"], r["text"], m.get("sender",""),
                                  m.get("timestamp",""), m.get("source",""))
                    st.success("Saved!")

# ─────────────────────────────────────────────────────────────────────────────
def page_analytics():
    auth = st.session_state.auth
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Communication patterns and team insights from your indexed chats.</div>', unsafe_allow_html=True)

    df = get_all_messages_df(auth["user_id"])

    if df.empty:
        st.info("No data yet. Import some chats first!")
        return

    # Top metrics
    total_msgs  = len(df)
    total_users = df["sender"].nunique() if "sender" in df.columns else 0
    total_src   = df["source"].nunique() if "source" in df.columns else 0
    total_ch    = df["channel"].nunique() if "channel" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{total_msgs:,}", "Total Messages"),
        (c2, str(total_users),  "Unique Senders"),
        (c3, str(total_src),    "Sources"),
        (c4, str(total_ch),     "Channels"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    figs = build_analytics(df)

    r1c1, r1c2 = st.columns(2)
    if "timeline" in figs:
        with r1c1:
            st.plotly_chart(figs["timeline"], use_container_width=True)
    if "users" in figs:
        with r1c2:
            st.plotly_chart(figs["users"], use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    if "heatmap" in figs:
        with r2c1:
            st.plotly_chart(figs["heatmap"], use_container_width=True)
    if "sources" in figs:
        with r2c2:
            st.plotly_chart(figs["sources"], use_container_width=True)

    # Top conversations table
    if "sender" in df.columns:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Top Senders</div>', unsafe_allow_html=True)
        top_df = df["sender"].value_counts().head(10).reset_index()
        top_df.columns = ["Sender", "Messages"]
        top_df["Share"] = (top_df["Messages"] / top_df["Messages"].sum() * 100).round(1).astype(str) + "%"
        st.dataframe(top_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
def page_bookmarks():
    auth    = st.session_state.auth
    api_key = st.session_state.get("runtime_api_key", "")

    st.markdown('<div class="section-title">🔖 Bookmarks & Saved Answers</div>', unsafe_allow_html=True)

    tab_bk, tab_ans = st.tabs(["  📌 Bookmarked Messages  ", "  🧠 Saved AI Answers  "])

    with tab_bk:
        bookmarks = get_bookmarks(auth["user_id"])
        if not bookmarks:
            st.info("No bookmarks yet. Save interesting messages while searching or asking.")
        else:
            st.markdown(f'<div style="margin-bottom:1rem;"><span class="neon-badge purple">{len(bookmarks)} saved</span></div>', unsafe_allow_html=True)
            for bk in bookmarks:
                bid, content, sender, ts, source, tag, created = bk
                st.markdown(f"""
                <div class="bookmark-card">
                    <div class="message-meta">{sender} · {ts} · {source.upper() if source else ''} · Saved {created[:10]}</div>
                    <div class="message-text">{content}</div>
                </div>""", unsafe_allow_html=True)
                if st.button("🗑 Remove", key=f"del_bk_{bid}"):
                    delete_bookmark(bid, auth["user_id"])
                    st.rerun()

    with tab_ans:
        answers = get_saved_answers(auth["user_id"])
        if not answers:
            st.info("No saved answers yet. Ask questions and click 'Save Answer'.")
        else:
            st.markdown(f'<div style="margin-bottom:1rem;"><span class="neon-badge">{len(answers)} saved</span></div>', unsafe_allow_html=True)
            for row in answers:
                aid, question, answer, created = row
                with st.expander(f"Q: {question[:80]}…  ({created[:10]})"):
                    st.markdown(f'<div class="ai-answer-box">{answer}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
def page_insights():
    auth    = st.session_state.auth
    api_key = st.session_state.get("runtime_api_key", "")

    st.markdown('<div class="section-title">⚡ AI Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Auto-generated intelligence from your conversations.</div>', unsafe_allow_html=True)

    if not api_key:
        st.warning("Add your OpenAI API key to use AI Insights.")
        return

    msgs = st.session_state.get("chat_messages", [])
    if not msgs:
        # Try to load from DB metadata
        chats = get_indexed_chats(auth["user_id"])
        if not chats:
            st.info("Import chats first to generate AI insights.")
            return
        st.info("Re-upload a chat file to generate fresh insights, or use cached data below.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Generate Summary", use_container_width=True) and msgs:
            with st.spinner("Generating summary…"):
                text = generate_summary(msgs, api_key)
            st.markdown(f'<div class="ai-answer-box">{text}</div>', unsafe_allow_html=True)

    with col2:
        if st.button("✅ Extract Action Items", use_container_width=True) and msgs:
            with st.spinner("Extracting action items…"):
                items = extract_action_items(msgs, api_key)
            st.markdown(f'<div class="ai-answer-box">{items}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Daily digest
    st.markdown('<div class="section-title" style="font-size:1.1rem;">📅 AI Daily Digest</div>', unsafe_allow_html=True)
    if st.button("Generate Today's Digest", use_container_width=True) and msgs:
        today_msgs = [m for m in msgs if datetime.date.today().isoformat() in m.get("timestamp","")]
        source_msgs = today_msgs if today_msgs else msgs[-30:]
        with st.spinner("Preparing digest…"):
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            txt = "\n".join(f"{m['sender']}: {m['text']}" for m in source_msgs[:60])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Create a concise daily digest of these chat messages. Include key topics, decisions, and people. Use emoji bullet points. Be engaging."},
                    {"role":"user","content":txt},
                ],
                max_tokens=500,
            )
        digest = resp.choices[0].message.content
        st.markdown(f'<div class="ai-answer-box">{digest}</div>', unsafe_allow_html=True)

    # Topic clusters
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.1rem;">🗂 Topic Detection</div>', unsafe_allow_html=True)
    if st.button("Detect Main Topics", use_container_width=True) and msgs:
        with st.spinner("Analyzing topics…"):
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            txt = "\n".join(m['text'] for m in msgs[:100])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Identify the top 6-8 main topics discussed. Return as JSON array of objects with 'topic' and 'description' keys only."},
                    {"role":"user","content":txt},
                ],
                max_tokens=300,
            )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```","",raw).strip()
        try:
            topics = json.loads(raw)
            cols = st.columns(2)
            for i, t in enumerate(topics):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="glass-card" style="margin-bottom:0.8rem;">
                        <div style="font-weight:700; color:var(--neon-cyan); margin-bottom:4px;">🏷 {t.get('topic','')}</div>
                        <div style="font-size:0.85rem; color:var(--text-muted);">{t.get('description','')}</div>
                    </div>""", unsafe_allow_html=True)
        except Exception:
            st.markdown(f'<div class="ai-answer-box">{raw}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.auth:
        render_auth()
        return

    render_sidebar()

    page = st.session_state.get("page", "ask")
    if page == "ask":
        page_ask()
    elif page == "import":
        page_import()
    elif page == "search":
        page_search()
    elif page == "analytics":
        page_analytics()
    elif page == "bookmarks":
        page_bookmarks()
    elif page == "insights":
        page_insights()
    else:
        page_ask()

if __name__ == "__main__":
    main()
