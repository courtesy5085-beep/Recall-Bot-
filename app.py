from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import time
import uuid
import zipfile
from collections import Counter
from contextlib import closing
from datetime import datetime, date
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Config & secrets
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="RecallBot · AI Memory for Team Chats",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.environ.get("RECALLBOT_DB", "recallbot.db")
CHROMA_PATH = os.environ.get("RECALLBOT_CHROMA", ".chroma")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-4o-mini"
MAX_UPLOAD_MB = 50


def get_openai_key() -> str | None:
    key = os.environ.get("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")
    if not key:
        try:
            key = st.secrets.get("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")  # type: ignore[attr-defined]
        except Exception:
            key = None
    return key


# ──────────────────────────────────────────────────────────────────────────────
# Styling — glassmorphism, gradients, dark/light
# ──────────────────────────────────────────────────────────────────────────────
def inject_css(dark: bool) -> None:
    if dark:
        bg = "radial-gradient(1200px 800px at 10% -10%, #1e1b4b 0%, transparent 60%), radial-gradient(900px 600px at 110% 10%, #0e7490 0%, transparent 55%), #0b1020"
        fg = "#e6e9f5"
        card = "rgba(255,255,255,0.06)"
        border = "rgba(255,255,255,0.12)"
        muted = "#9aa3c7"
    else:
        bg = "radial-gradient(1200px 800px at 10% -10%, #e0e7ff 0%, transparent 60%), radial-gradient(900px 600px at 110% 10%, #cffafe 0%, transparent 55%), #f7f8ff"
        fg = "#0b1020"
        card = "rgba(255,255,255,0.7)"
        border = "rgba(15,23,42,0.08)"
        muted = "#475569"

    st.markdown(
        f"""
    <style>
      .stApp {{
        background: {bg};
        color: {fg};
      }}
      .block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
      h1, h2, h3, h4 {{ color: {fg}; letter-spacing: -0.01em; }}
      .rb-hero {{
        background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 50%, #22d3ee 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        font-weight: 800; font-size: 2.2rem; line-height: 1.1;
      }}
      .rb-sub {{ color: {muted}; margin-top: -.25rem; }}
      .rb-card {{
        background: {card};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {border};
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 10px 40px rgba(2, 6, 23, 0.18);
        transition: transform .25s ease, box-shadow .25s ease;
      }}
      .rb-card:hover {{ transform: translateY(-2px); box-shadow: 0 18px 50px rgba(2,6,23,.28); }}
      .rb-metric-label {{ color: {muted}; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }}
      .rb-metric-value {{ font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg,#a78bfa,#22d3ee); -webkit-background-clip: text; color: transparent; }}
      .rb-chip {{ display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.72rem; border:1px solid {border}; color:{muted}; margin-right:.35rem; }}
      .rb-msg-user, .rb-msg-ai {{
        border-radius: 16px; padding: .85rem 1rem; margin: .35rem 0;
        border: 1px solid {border}; background: {card}; backdrop-filter: blur(8px);
      }}
      .rb-msg-ai {{ background: linear-gradient(135deg, rgba(124,58,237,.10), rgba(6,182,212,.10)); }}
      .rb-cite {{ font-size:.82rem; color:{muted}; border-left:3px solid #7c3aed; padding:.4rem .6rem; margin:.3rem 0; background: {card}; border-radius: 8px; }}
      .stButton>button, .stDownloadButton>button {{
        border-radius: 12px !important;
        border: 1px solid {border} !important;
        background: linear-gradient(135deg,#7c3aed,#06b6d4) !important;
        color: white !important; font-weight:600 !important;
        transition: transform .15s ease, box-shadow .15s ease;
      }}
      .stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 10px 25px rgba(124,58,237,.35) !important; }}
      .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
        border-radius: 12px !important;
      }}
      section[data-testid="stSidebar"] {{
        background: {card}; backdrop-filter: blur(16px); border-right: 1px solid {border};
      }}
      .rb-skeleton {{
        height: 14px; border-radius: 8px;
        background: linear-gradient(90deg, {border}, {card}, {border});
        background-size: 200% 100%; animation: rb-shimmer 1.4s infinite;
        margin: .35rem 0;
      }}
      @keyframes rb-shimmer {{ 0%{{background-position:200% 0}} 100%{{background-position:-200% 0}} }}
      .rb-typing::after {{ content:"▍"; animation: rb-blink 1s steps(2) infinite; }}
      @keyframes rb-blink {{ 50% {{ opacity: 0; }} }}
      [data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed {border} !important; border-radius: 16px !important;
        background: {card} !important;
      }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SQLite — users, bookmarks, saved queries, tags
# ──────────────────────────────────────────────────────────────────────────────
def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    with closing(db_conn()) as c, c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                pwhash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bookmarks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                msg_id TEXT NOT NULL,
                sender TEXT, ts TEXT, text TEXT, source TEXT,
                tag TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS saved_queries(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    ).hex()


def create_user(email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Invalid email."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = uuid.uuid4().hex
    pwhash = hash_password(password, salt)
    try:
        with closing(db_conn()) as c, c:
            c.execute(
                "INSERT INTO users(email,salt,pwhash,created_at) VALUES(?,?,?,?)",
                (email, salt, pwhash, datetime.utcnow().isoformat()),
            )
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def verify_user(email: str, password: str) -> dict | None:
    email = email.strip().lower()
    with closing(db_conn()) as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not row:
            return None
        if hash_password(password, row["salt"]) == row["pwhash"]:
            return {"id": row["id"], "email": row["email"]}
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Parsers — WhatsApp (.txt), Slack (.json / .zip)
# ──────────────────────────────────────────────────────────────────────────────
WA_LINE = re.compile(
    r"^\[?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\]?\s*[-–]?\s*([^:]{1,80}):\s(.*)$"
)
MEDIA_TOKENS = ("<Media omitted>", "image omitted", "video omitted", "sticker omitted",
                "audio omitted", "GIF omitted", "document omitted")


def _parse_wa_dt(d: str, t: str) -> str | None:
    for fmt in (
        "%d/%m/%Y %H:%M", "%d/%m/%y %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M",
        "%d/%m/%Y %I:%M %p", "%d/%m/%y %I:%M %p", "%m/%d/%Y %I:%M %p", "%m/%d/%y %I:%M %p",
        "%d.%m.%Y %H:%M", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(f"{d} {t.upper().replace('  ', ' ').strip()}", fmt).isoformat()
        except ValueError:
            continue
    return None


def parse_whatsapp(text: str, source: str) -> list[dict]:
    messages: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.replace("\u202f", " ").replace("\u200e", "").strip()
        if not line:
            continue
        m = WA_LINE.match(line)
        if m:
            if current:
                messages.append(current)
            d, t, sender, msg = m.groups()
            if any(tok.lower() in msg.lower() for tok in MEDIA_TOKENS):
                current = None
                continue
            current = {
                "id": str(uuid.uuid4()),
                "ts": _parse_wa_dt(d, t) or "",
                "sender": sender.strip()[:80],
                "text": msg.strip(),
                "source": source,
            }
        else:
            if current:
                current["text"] += " " + line
    if current:
        messages.append(current)
    return [m for m in messages if m["text"]]


def parse_slack_json(data: Any, source: str) -> list[dict]:
    messages: list[dict] = []
    items = data if isinstance(data, list) else data.get("messages", [])
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("subtype") in {"channel_join", "channel_leave", "bot_message"}:
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue
        ts_raw = it.get("ts") or it.get("timestamp")
        try:
            ts_iso = datetime.utcfromtimestamp(float(ts_raw)).isoformat() if ts_raw else ""
        except (TypeError, ValueError):
            ts_iso = ""
        sender = (
            it.get("user_profile", {}).get("real_name")
            or it.get("user_profile", {}).get("display_name")
            or it.get("user") or it.get("username") or "Unknown"
        )
        messages.append({
            "id": str(uuid.uuid4()),
            "ts": ts_iso,
            "sender": str(sender)[:80],
            "text": text,
            "source": source,
        })
    return messages


def parse_slack_zip(file_bytes: bytes, source: str) -> list[dict]:
    out: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            if not name.endswith(".json") or name.endswith("/"):
                continue
            if any(skip in name for skip in ("users.json", "channels.json", "integration_logs.json")):
                continue
            try:
                with zf.open(name) as f:
                    data = json.loads(f.read().decode("utf-8", errors="ignore"))
                out.extend(parse_slack_json(data, source=f"{source}:{name}"))
            except Exception:
                continue
    return out


={"hnsw:space": "cosine"}
    )


def chunk_messages(messages: list[dict], chunk_chars: int = 900, overlap: int = 120) -> list[dict]:
    """Group consecutive messages into ~chunk_chars windows, preserving metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)

    
        import numpy as np
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> np.ndarray:
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)

def search(query: str, vectors: np.ndarray, docs: list[dict], k=8):
    q = embed([query])[0]
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-9)
    idx = np.argsort(-sims)[:k]
    return [docs[i] | {"score": float(sims[i])} for i in idx]



# ──────────────────────────────────────────────────────────────────────────────
# OpenAI streaming
# ──────────────────────────────────────────────────────────────────────────────
def openai_client():
    from openai import OpenAI
    key = get_openai_key()
    if not key:
        return None
    return OpenAI(api_key=key)


SYSTEM_PROMPT = """You are RecallBot, an AI assistant that answers questions strictly from the user's imported chat history.

Rules:
- Use ONLY the provided context messages. If the answer isn't in the context, say so honestly.
- Always cite the sender and timestamp of the messages you rely on, e.g. (Ahmed, 2024-03-12 14:02).
- Be concise, structured, and accurate. Use bullet points for action items or lists.
- Never invent senders, dates, or quotes."""


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        parts.append(
            f"[Source {i}] senders={m.get('senders','?')} | "
            f"from={m.get('first_ts','?')} to={m.get('last_ts','?')} | "
            f"file={m.get('source','?')}\n{h['text']}"
        )
    return "\n\n---\n\n".join(parts)


def stream_answer(question: str, hits: list[dict]):
    client = openai_client()
    context = build_context(hits) if hits else "(no context available)"
    user_prompt = f"Question: {question}\n\nContext:\n{context}"

    if client is None:
        # Graceful fallback if no API key is configured.
        msg = ("⚠️ `OPENAI_API_KEY` is not configured. Showing retrieved context only.\n\n"
               "**Top matches:**\n\n")
        for h in hits[:3]:
            msg += f"- _{h['meta'].get('senders','?')}_ — {h['text'][:200]}…\n"
        for ch in msg.split(" "):
            yield ch + " "
            time.sleep(0.005)
        return

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.2,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except (IndexError, AttributeError):
                delta = ""
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n⚠️ Error contacting OpenAI: `{e}`"


# ──────────────────────────────────────────────────────────────────────────────
# Auth views
# ──────────────────────────────────────────────────────────────────────────────
def view_auth() -> None:
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="rb-hero">🧠 RecallBot</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="rb-sub">Your team\'s collective memory — searchable, citable, instant.</p>',
            unsafe_allow_html=True,
        )
        st.write("")
        with st.container():
            st.markdown('<div class="rb-card">', unsafe_allow_html=True)
            tab_in, tab_up = st.tabs(["🔐 Log in", "✨ Sign up"])
            with tab_in:
                with st.form("login"):
                    email = st.text_input("Email", placeholder="you@team.com")
                    pw = st.text_input("Password", type="password")
                    ok = st.form_submit_button("Log in", use_container_width=True)
                if ok:
                    u = verify_user(email, pw)
                    if u:
                        st.session_state.user = u
                        st.toast("Welcome back 👋", icon="🎉")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
            with tab_up:
                with st.form("signup"):
                    email = st.text_input("Email", key="su_email", placeholder="you@team.com")
                    pw = st.text_input("Password", type="password", key="su_pw",
                                       help="At least 6 characters.")
                    ok = st.form_submit_button("Create account", use_container_width=True)
                if ok:
                    success, msg = create_user(email, pw)
                    (st.success if success else st.error)(msg)
            st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar(user: dict) -> str:
    with st.sidebar:
        st.markdown(f"### 🧠 RecallBot")
        st.caption(f"Signed in as **{user['email']}**")

        st.session_state.dark = st.toggle("🌙 Dark mode", value=st.session_state.get("dark", True))

        nav = st.radio("Navigate", ["💬 Chat", "📥 Import", "🔎 Search",
                                     "📊 Analytics", "🔖 Memory", "⚙️ Settings"],
                       label_visibility="collapsed")

        st.divider()
        st.markdown("#### 📥 Quick upload")
        files = st.file_uploader(
            "Drop WhatsApp .txt or Slack .json / .zip",
            type=["txt", "json", "zip"],
            accept_multiple_files=True,
            key="quick_upload",
            label_visibility="collapsed",
        )
        if files and st.button("⚡ Index files", use_container_width=True):
            handle_uploads(user, files)

        st.divider()
        coll = user_collection(user["id"])
        try:
            count = coll.count()
        except Exception:
            count = 0
        st.markdown("#### 💾 Storage")
        st.progress(min(1.0, count / 5000), text=f"{count:,} chunks indexed")

        st.divider()
        if st.button("🚪 Log out", use_container_width=True):
            for k in ["user", "messages", "raw_messages"]:
                st.session_state.pop(k, None)
            st.rerun()
    return nav


# ──────────────────────────────────────────────────────────────────────────────
# Upload handler
# ──────────────────────────────────────────────────────────────────────────────
def handle_uploads(user: dict, files: Iterable) -> None:
    total_msgs = 0
    total_chunks = 0
    progress = st.progress(0.0, text="Parsing…")
    for f in files:
        try:
            size_mb = len(f.getbuffer()) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.warning(f"`{f.name}` is larger than {MAX_UPLOAD_MB} MB — skipped.")
                continue
            name = f.name.lower()
            data = f.getvalue()
            if name.endswith(".txt"):
                msgs = parse_whatsapp(data.decode("utf-8", errors="ignore"), source=f.name)
            elif name.endswith(".json"):
                msgs = parse_slack_json(json.loads(data.decode("utf-8", errors="ignore")), source=f.name)
            elif name.endswith(".zip"):
                msgs = parse_slack_zip(data, source=f.name)
            else:
                st.warning(f"Unsupported file: {f.name}")
                continue
            if not msgs:
                st.warning(f"No messages parsed from `{f.name}`.")
                continue
            total_msgs += len(msgs)
            total_chunks += index_messages(user["id"], msgs, progress=progress)
            st.toast(f"Indexed {len(msgs):,} messages from {f.name}", icon="✅")
        except Exception as e:
            st.error(f"Failed to import `{f.name}`: {e}")
    progress.empty()
    if total_msgs:
        st.success(f"Imported **{total_msgs:,}** messages → **{total_chunks:,}** vector chunks.")


# ──────────────────────────────────────────────────────────────────────────────
# Views
# ──────────────────────────────────────────────────────────────────────────────
def view_chat(user: dict) -> None:
    st.markdown('<div class="rb-hero">💬 Ask your chat history</div>', unsafe_allow_html=True)
    st.markdown('<p class="rb-sub">Streaming answers, grounded in your messages, with citations.</p>',
                unsafe_allow_html=True)

    msgs = st.session_state.setdefault("messages", [])

    for m in msgs:
        css = "rb-msg-user" if m["role"] == "user" else "rb-msg-ai"
        with st.chat_message(m["role"]):
            st.markdown(f'<div class="{css}">{m["content"]}</div>', unsafe_allow_html=True)
            if m["role"] == "assistant" and m.get("citations"):
                with st.expander(f"📎 {len(m['citations'])} citation(s)"):
                    for c in m["citations"]:
                        meta = c["meta"]
                        st.markdown(
                            f'<div class="rb-cite"><b>{meta.get("senders","?")}</b> · '
                            f'{meta.get("first_ts","?")} · <i>{meta.get("source","?")}</i><br>'
                            f'{c["text"][:600]}{"…" if len(c["text"])>600 else ""}</div>',
                            unsafe_allow_html=True,
                        )

    prompt = st.chat_input("Ask anything about your imported chats…")
    if not prompt:
        st.info("💡 Try: *“Summarize last week’s discussion”* or *“Who mentioned the API bug?”*")
        return

    msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="rb-msg-user">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="rb-msg-ai"><div class="rb-skeleton" style="width:80%"></div>'
            '<div class="rb-skeleton" style="width:60%"></div>'
            '<div class="rb-skeleton" style="width:70%"></div></div>',
            unsafe_allow_html=True,
        )
        hits = semantic_search(user["id"], prompt, k=6)
        acc = ""
        for tok in stream_answer(prompt, hits):
            acc += tok
            placeholder.markdown(
                f'<div class="rb-msg-ai rb-typing">{acc}</div>',
                unsafe_allow_html=True,
            )
        placeholder.markdown(f'<div class="rb-msg-ai">{acc}</div>', unsafe_allow_html=True)
        if hits:
            with st.expander(f"📎 {len(hits)} citation(s)"):
                for c in hits:
                    meta = c["meta"]
                    st.markdown(
                        f'<div class="rb-cite"><b>{meta.get("senders","?")}</b> · '
                        f'{meta.get("first_ts","?")} · <i>{meta.get("source","?")}</i><br>'
                        f'{c["text"][:600]}{"…" if len(c["text"])>600 else ""}</div>',
                        unsafe_allow_html=True,
                    )

    msgs.append({"role": "assistant", "content": acc, "citations": hits})


def view_import(user: dict) -> None:
    st.markdown('<div class="rb-hero">📥 Import conversations</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rb-sub">Drop WhatsApp exports (.txt) or Slack exports (.json / .zip). '
        'Files are parsed, chunked, embedded, and stored locally.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    files = st.file_uploader(
        "Drop files here",
        type=["txt", "json", "zip"],
        accept_multiple_files=True,
        key="main_upload",
    )
    if files and st.button("⚡ Parse & index all", use_container_width=True):
        handle_uploads(user, files)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="rb-card"><b>📱 WhatsApp</b><br>'
                    'In WhatsApp → Chat → ⋮ → <i>More → Export chat → Without media</i>.'
                    '</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="rb-card"><b>💼 Slack</b><br>'
                    'Workspace owners can export from '
                    '<i>Settings → Import/Export Data → Export</i>. Upload the .zip.'
                    '</div>', unsafe_allow_html=True)


def view_search(user: dict) -> None:
    st.markdown('<div class="rb-hero">🔎 Advanced search</div>', unsafe_allow_html=True)
    st.markdown('<p class="rb-sub">Semantic + keyword search with date and sender filters.</p>',
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="rb-card">', unsafe_allow_html=True)
        q = st.text_input("Query", placeholder="e.g. budget approval")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1: sender = st.text_input("Sender contains", "")
        with c2: dfrom = st.date_input("From", value=None)
        with c3: dto = st.date_input("To", value=None)
        with c4: mode = st.selectbox("Mode", ["Semantic", "Keyword"])
        go = st.button("Search", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if go and q.strip():
        with st.spinner("Searching…"):
            if mode == "Semantic":
                hits = semantic_search(user["id"], q, k=20,
                                       sender=sender or None,
                                       date_from=dfrom or None,
                                       date_to=dto or None)
            else:
                hits = keyword_search(user["id"], q, sender=sender or None,
                                      date_from=dfrom or None, date_to=dto or None)
        if not hits:
            st.info("No results.")
            return
        st.success(f"{len(hits)} result(s)")
        df = pd.DataFrame([{
            "score": round(h.get("score", 0), 3),
            "senders": h["meta"].get("senders"),
            "first_ts": h["meta"].get("first_ts"),
            "source": h["meta"].get("source"),
            "text": h["text"],
        } for h in hits])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="recallbot_results.csv", mime="text/csv")


def keyword_search(user_id: int, q: str, sender=None,
                   date_from=None, date_to=None) -> list[dict]:
    coll = user_collection(user_id)
    try:
        got = coll.get(include=["documents", "metadatas"])
    except Exception:
        return []
    out: list[dict] = []
    ql = q.lower()
    for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
        if ql not in (doc or "").lower():
            continue
        meta = meta or {}
        if sender and sender.lower() not in (meta.get("senders") or "").lower():
            continue
        ts_str = meta.get("first_ts") or ""
        try:
            ts_date = datetime.fromisoformat(ts_str).date() if ts_str else None
        except ValueError:
            ts_date = None
        if date_from and ts_date and ts_date < date_from: continue
        if date_to and ts_date and ts_date > date_to: continue
        out.append({"text": doc, "meta": meta, "score": 1.0})
    return out[:200]


STOPWORDS = set("""
the a an and or but if then so of to in on at by for with from as is are was were be been being
this that these those it its i you he she we they them us our your their my me him her his hers
not no do does did done has have had will would should could can may might just about into out
up down over under more most some any all each every other another such same than too very
""".split())


def view_analytics(user: dict) -> None:
    st.markdown('<div class="rb-hero">📊 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="rb-sub">Activity, top voices, and hot topics across your archive.</p>',
                unsafe_allow_html=True)

    raw = st.session_state.get("raw_messages", [])
    if not raw:
        coll = user_collection(user["id"])
        try:
            got = coll.get(include=["metadatas", "documents"])
            raw = []
            for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                raw.append({
                    "ts": (meta or {}).get("first_ts", ""),
                    "sender": ((meta or {}).get("senders") or "Unknown").split(",")[0].strip(),
                    "text": doc or "",
                    "source": (meta or {}).get("source", ""),
                })
        except Exception:
            raw = []

    if not raw:
        st.info("Import some chats to see analytics.")
        return

    df = pd.DataFrame(raw)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        (c1, c2, c3, c4),
        ("Messages", "Unique senders", "Sources", "Days covered"),
        (len(df), df["sender"].nunique(), df["source"].nunique(),
         (df["ts"].max() - df["ts"].min()).days if len(df) else 0),
    ):
        col.markdown(
            f'<div class="rb-card"><div class="rb-metric-label">{label}</div>'
            f'<div class="rb-metric-value">{val:,}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    cA, cB = st.columns(2)
    with cA:
        top = df["sender"].value_counts().head(10).reset_index()
        top.columns = ["sender", "count"]
        fig = px.bar(top, x="count", y="sender", orientation="h",
                     title="Most active users",
                     color="count", color_continuous_scale="Plasma")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        tl = df.groupby(df["ts"].dt.date).size().reset_index(name="count")
        fig = px.area(tl, x="ts", y="count", title="Conversation timeline",
                      color_discrete_sequence=["#22d3ee"])
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.day_name()
    heat = df.groupby(["dow", "hour"]).size().reset_index(name="count")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig = px.density_heatmap(heat, x="hour", y="dow", z="count",
                             category_orders={"dow": order},
                             title="Activity heatmap (hour × day)",
                             color_continuous_scale="Viridis")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    words = Counter()
    for t in df["text"].astype(str):
        for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", t.lower()):
            if w not in STOPWORDS:
                words[w] += 1
    top_words = pd.DataFrame(words.most_common(20), columns=["word", "count"])
    if not top_words.empty:
        fig = px.bar(top_words, x="word", y="count", title="Top keywords",
                     color="count", color_continuous_scale="Magma")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


def view_memory(user: dict) -> None:
    st.markdown('<div class="rb-hero">🔖 Memory</div>', unsafe_allow_html=True)
    st.markdown('<p class="rb-sub">Bookmarks, saved queries, and AI summaries.</p>',
                unsafe_allow_html=True)

    tab_b, tab_q, tab_s = st.tabs(["📌 Bookmarks", "💡 Saved queries", "🧾 AI summary"])

    with tab_b:
        with closing(db_conn()) as c:
            rows = c.execute(
                "SELECT * FROM bookmarks WHERE user_id=? ORDER BY id DESC", (user["id"],)
            ).fetchall()
        if not rows:
            st.info("Pin messages from the Search tab — feature: paste below.")
        for r in rows:
            st.markdown(
                f'<div class="rb-card"><span class="rb-chip">{r["tag"] or "note"}</span>'
                f'<b>{r["sender"] or "?"}</b> · {r["ts"] or "?"}<br>{r["text"]}</div>',
                unsafe_allow_html=True,
            )

        with st.form("add_bm"):
            txt = st.text_area("Pin a note or message")
            tag = st.text_input("Tag", value="note")
            if st.form_submit_button("📌 Pin"):
                if txt.strip():
                    with closing(db_conn()) as c, c:
                        c.execute(
                            "INSERT INTO bookmarks(user_id,msg_id,sender,ts,text,source,tag,created_at) "
                            "VALUES(?,?,?,?,?,?,?,?)",
                            (user["id"], str(uuid.uuid4()), user["email"],
                             datetime.utcnow().isoformat(), txt.strip(), "manual", tag,
                             datetime.utcnow().isoformat()),
                        )
                    st.toast("Pinned", icon="📌"); st.rerun()

    with tab_q:
        with closing(db_conn()) as c:
            rows = c.execute(
                "SELECT * FROM saved_queries WHERE user_id=? ORDER BY id DESC", (user["id"],)
            ).fetchall()
        for r in rows:
            st.markdown(f'<div class="rb-card">💡 {r["query"]}</div>', unsafe_allow_html=True)
        with st.form("add_q"):
            q = st.text_input("Save a useful query")
            if st.form_submit_button("💾 Save"):
                if q.strip():
                    with closing(db_conn()) as c, c:
                        c.execute(
                            "INSERT INTO saved_queries(user_id,query,created_at) VALUES(?,?,?)",
                            (user["id"], q.strip(), datetime.utcnow().isoformat()),
                        )
                    st.toast("Saved", icon="💾"); st.rerun()

    with tab_s:
        topic = st.text_input("Topic to summarize", placeholder="e.g. Q4 planning")
        if st.button("🪄 Generate summary") and topic.strip():
            with st.spinner("Thinking…"):
                hits = semantic_search(user["id"], topic, k=10)
                placeholder = st.empty()
                acc = ""
                for tok in stream_answer(f"Summarize the discussion about: {topic}", hits):
                    acc += tok
                    placeholder.markdown(f'<div class="rb-msg-ai rb-typing">{acc}</div>',
                                         unsafe_allow_html=True)
                placeholder.markdown(f'<div class="rb-msg-ai">{acc}</div>', unsafe_allow_html=True)


def view_settings(user: dict) -> None:
    st.markdown('<div class="rb-hero">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<p class="rb-sub">Manage your workspace and data.</p>', unsafe_allow_html=True)

    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    st.markdown(f"**Account:** {user['email']}")
    st.markdown(f"**OpenAI key:** {'✅ configured' if get_openai_key() else '⚠️ not set — set OPENAI_API_KEY'}")
    st.markdown(f"**Embedding model:** `{EMBED_MODEL}`")
    st.markdown(f"**Chat model:** `{CHAT_MODEL}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    st.warning("Danger zone")
    if st.button("🗑️ Delete all my indexed data"):
        try:
            client = get_chroma()
            client.delete_collection(f"user_{user['id']}")
        except Exception:
            pass
        st.session_state.pop("raw_messages", None)
        st.session_state.pop("messages", None)
        st.toast("Wiped your vector store", icon="🗑️")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    db_init()
    if "dark" not in st.session_state:
        st.session_state.dark = True
    inject_css(st.session_state.dark)

    if "user" not in st.session_state:
        view_auth()
        return

    user = st.session_state.user
    nav = render_sidebar(user)

    if nav.endswith("Chat"):       view_chat(user)
    elif nav.endswith("Import"):   view_import(user)
    elif nav.endswith("Search"):   view_search(user)
    elif nav.endswith("Analytics"):view_analytics(user)
    elif nav.endswith("Memory"):   view_memory(user)
    elif nav.endswith("Settings"): view_settings(user)


if __name__ == "__main__":
    main()
