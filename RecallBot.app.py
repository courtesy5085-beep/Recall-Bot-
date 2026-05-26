"""
RecallBot — AI memory for team chats.
Upload WhatsApp/Slack exports, index them, and ask questions with cited answers.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from typing import Generator

import chromadb
import streamlit as st
import tiktoken
from dateutil import parser as dateutil_parser
from fastembed import TextEmbedding
from fpdf import FPDF
from openai import OpenAI

# ─── Constants ────────────────────────────────────────────────────────────────

CHROMA_PATH = "./chroma_db"
DB_PATH = "recallbot.db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_BATCH = 128
TOP_K = 7
SIMILARITY_THRESHOLD = 0.3
MODEL = "gpt-4o-mini"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

SAMPLE_MESSAGES = [
    {"date": "12 Oct 2024", "sender": "Sara", "text": "We've decided to go with the new onboarding flow starting November 1st."},
    {"date": "12 Oct 2024", "sender": "Alex", "text": "Action item: Alex to update the landing page copy by Oct 20th."},
    {"date": "13 Oct 2024", "sender": "Jordan", "text": "Should we integrate Stripe or Paddle for payments? Still unresolved."},
    {"date": "14 Oct 2024", "sender": "Sara", "text": "Key decision: The API rate limit will be 1000 req/min for free tier."},
    {"date": "15 Oct 2024", "sender": "Alex", "text": "Reminder: team sync every Monday at 10 AM. Jordan to send calendar invite."},
]

# ─── CSS ──────────────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --base: #0A0A0B;
    --surface: #111114;
    --surface2: #16161A;
    --accent: #6366F1;
    --accent-hover: #4F52D9;
    --text: #E5E7EB;
    --muted: #9CA3AF;
    --border: #1F2937;
    --success: #10B981;
    --danger: #EF4444;
    --radius: 10px;
    --radius-sm: 6px;
}

* { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
    background-color: var(--base) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Input fields */
input, textarea, [data-baseweb="input"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    min-height: 44px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0 18px !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary buttons */
.stButton.secondary > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.stButton.secondary > button:hover {
    background: var(--border) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 12px !important;
    animation: fadeIn 0.3s ease;
}
[data-testid="stChatMessageContent"] * { color: var(--text) !important; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
}

/* Cards */
.rb-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.rb-card:hover { border-color: var(--accent); }

/* Source chip */
.rb-source-chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin: 3px 3px 3px 0;
    cursor: pointer;
    transition: border-color 0.2s, color 0.2s;
}
.rb-source-chip:hover { border-color: var(--accent); color: var(--accent); }

/* Progress bar shimmer */
.shimmer-container {
    width: 100%;
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
    margin: 12px 0;
}
.shimmer-bar {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--accent) 0%, #818CF8 50%, var(--accent) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}
@keyframes shimmer {
    0% { background-position: 200% center; }
    100% { background-position: -200% center; }
}

/* Label / heading */
.rb-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}

.rb-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}

.rb-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 4px;
}

/* Highlighted term */
mark {
    background: rgba(99,102,241,0.3) !important;
    color: var(--text) !important;
    border-radius: 2px;
    padding: 0 2px;
}

/* Divider */
.rb-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Selectbox */
[data-baseweb="select"] {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--muted) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Mobile responsive */
@media (max-width: 768px) {
    .rb-right-panel { display: none !important; }
    [data-testid="stSidebar"] { width: 100% !important; }
}

/* Success/danger badges */
.rb-badge-success { color: var(--success); font-size: 0.8rem; }
.rb-badge-danger  { color: var(--danger);  font-size: 0.8rem; }

/* Recent questions list */
.rb-recent-q {
    font-size: 0.82rem;
    color: var(--muted);
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.rb-recent-q:hover { background: var(--surface2); color: var(--text); }

.stTextInput > div > div > input {
    min-height: 44px !important;
}
</style>
"""

# ─── Database helpers ──────────────────────────────────────────────────────────

def init_db() -> None:
    """Create SQLite tables for feedback and question history."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                question TEXT,
                answer TEXT,
                thumbs INTEGER,
                workspace TEXT,
                created_at TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                workspace TEXT,
                created_at TEXT
            )
        """)
        con.commit()


def save_question(question: str, workspace: str) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO questions (question, workspace, created_at) VALUES (?,?,?)",
            (question, workspace, datetime.utcnow().isoformat()),
        )
        con.commit()


def get_recent_questions(workspace: str, limit: int = 10) -> list[str]:
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT question FROM questions WHERE workspace=? ORDER BY id DESC LIMIT ?",
            (workspace, limit),
        ).fetchall()
    return [r[0] for r in rows]


def save_feedback(msg_id: str, question: str, answer: str, thumbs: int, workspace: str) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT OR REPLACE INTO feedback (id,question,answer,thumbs,workspace,created_at) VALUES (?,?,?,?,?,?)",
            (msg_id, question, answer, thumbs, workspace, datetime.utcnow().isoformat()),
        )
        con.commit()


# ─── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_chroma_client() -> chromadb.Client:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


@st.cache_resource
def get_embedding_model() -> TextEmbedding:
    """Load fastembed BAAI/bge-small-en-v1.5."""
    return TextEmbedding(model_name=EMBED_MODEL_NAME)


def get_openai_client() -> OpenAI:
    """Return an OpenAI client using secrets."""
    key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not key:
        st.error("OpenAI API key not found. Add OPENAI_API_KEY to st.secrets or environment.")
        st.stop()
    return OpenAI(api_key=key)


# ─── Parsing ──────────────────────────────────────────────────────────────────

def detect_format(content: str) -> str:
    """Detect whether the file is WhatsApp or Slack JSON."""
    stripped = content.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            json.loads(stripped)
            return "slack"
        except Exception:
            pass
    return "whatsapp"


def parse_whatsapp(content: str) -> list[dict]:
    """Parse WhatsApp export: [dd/mm/yy, hh:mm] Sender: Message."""
    pattern = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s+([^:]+):\s+(.*)"
    )
    messages = []
    current = None
    for line in content.splitlines():
        m = pattern.match(line.strip())
        if m:
            if current:
                messages.append(current)
            date_str, time_str, sender, text = m.groups()
            try:
                dt = dateutil_parser.parse(f"{date_str} {time_str}", dayfirst=True)
                date_fmt = dt.strftime("%d %b %Y")
            except Exception:
                date_fmt = date_str
            current = {"date": date_fmt, "sender": sender.strip(), "text": text.strip()}
        elif current and line.strip():
            current["text"] += " " + line.strip()
    if current:
        messages.append(current)
    return messages


def parse_slack(content: str) -> list[dict]:
    """Parse Slack JSON export."""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # channel export (dict of lists)
            msgs = []
            for v in data.values():
                if isinstance(v, list):
                    msgs.extend(v)
        elif isinstance(data, list):
            msgs = data
        else:
            return []
        messages = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            text = m.get("text", "").strip()
            sender = (
                m.get("user_profile", {}).get("display_name")
                or m.get("username")
                or m.get("user", "Unknown")
            )
            ts = m.get("ts", "")
            try:
                dt = datetime.fromtimestamp(float(ts))
                date_fmt = dt.strftime("%d %b %Y")
            except Exception:
                date_fmt = "Unknown"
            if text:
                messages.append({"date": date_fmt, "sender": sender, "text": text})
        return messages
    except Exception:
        return []


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_messages(messages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Convert messages to token-chunked documents with metadata.
    Returns list of {text, date, sender, chunk_id}.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    token_buf: list[int] = []
    meta_buf: list[dict] = []

    def flush(toks: list[int], metas: list[dict]) -> dict:
        text = enc.decode(toks)
        # use the first message's metadata for the chunk
        m = metas[0] if metas else {}
        return {
            "text": text,
            "date": m.get("date", ""),
            "sender": m.get("sender", ""),
            "chunk_id": str(uuid.uuid4()),
        }

    for msg in messages:
        msg_text = f"[{msg['date']}] {msg['sender']}: {msg['text']}"
        toks = enc.encode(msg_text)
        for tok in toks:
            token_buf.append(tok)
            meta_buf.append(msg)
            if len(token_buf) >= chunk_size:
                chunks.append(flush(token_buf, meta_buf))
                # keep overlap
                token_buf = token_buf[-overlap:]
                meta_buf = meta_buf[-overlap:]

    if token_buf:
        chunks.append(flush(token_buf, meta_buf))

    return chunks


# ─── Embedding & Indexing ─────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using fastembed."""
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def get_or_create_collection(workspace: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for the workspace."""
    client = get_chroma_client()
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", workspace)[:60] or "default"
    return client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_messages(messages: list[dict], workspace: str, progress_placeholder) -> dict:
    """
    Chunk, embed, and store messages in ChromaDB.
    Returns stats dict.
    """
    start = time.time()
    n_messages = len(messages)

    chunks = chunk_messages(messages)
    n_chunks = len(chunks)

    collection = get_or_create_collection(workspace)

    # Embed in batches
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"date": c["date"], "sender": c["sender"]} for c in chunks]

    done = 0
    for batch_start in range(0, n_chunks, EMBED_BATCH):
        batch_end = min(batch_start + EMBED_BATCH, n_chunks)
        batch_texts = texts[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]
        batch_meta = metadatas[batch_start:batch_end]
        batch_emb = embed_texts(batch_texts)

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_emb,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        done = batch_end
        progress_placeholder.markdown(
            f"<div class='shimmer-container'><div class='shimmer-bar' style='width:{int(done/n_chunks*100)}%'></div></div>"
            f"<span style='color:var(--muted);font-size:0.82rem'>Indexing {n_messages:,} messages... {done:,}/{n_chunks:,} chunks</span>",
            unsafe_allow_html=True,
        )

    elapsed = round(time.time() - start, 1)
    return {"messages": n_messages, "chunks": n_chunks, "seconds": elapsed}


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, workspace: str) -> list[dict]:
    """Embed query, query ChromaDB, return top-K results."""
    collection = get_or_create_collection(workspace)
    q_emb = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = 1 - dist  # cosine distance → similarity
        chunks.append({
            "text": doc,
            "date": meta.get("date", ""),
            "sender": meta.get("sender", ""),
            "similarity": round(similarity, 4),
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c['date']}][{c['sender']}]: {c['text']}")
    return "\n\n".join(lines)


def stream_answer(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    """Stream a GPT-4o-mini answer given retrieved chunks."""
    context = build_context(chunks)
    system = (
        "You are RecallBot. Answer ONLY using the [CONTEXT] provided. "
        "Cite sources inline as [Date][Sender]. "
        "At the end of your answer, add a 'Sources:' section listing each source as [Date, Sender]. "
        "If the answer is not found in the context, say exactly: 'I couldn't find that in the chat history.'"
    )
    user_msg = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{query}"
    client = get_openai_client()
    try:
        with client.chat.completions.stream(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=900,
            temperature=0.2,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\n_Error communicating with OpenAI: {e}_"


# ─── Summarize week ───────────────────────────────────────────────────────────

def summarize_week(workspace: str) -> dict:
    """Retrieve last-7-days messages and summarize into JSON structure."""
    collection = get_or_create_collection(workspace)
    week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%d %b %Y")

    # Retrieve a broad sample (no date filter in ChromaDB easily; use large n_results)
    results = collection.get(limit=500, include=["documents", "metadatas"])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    context_lines = []
    for doc, meta in zip(docs, metas):
        context_lines.append(f"[{meta.get('date','')}][{meta.get('sender','')}]: {doc}")

    context = "\n".join(context_lines[:300])  # cap to avoid huge prompt

    client = get_openai_client()
    prompt = (
        "You are RecallBot summarizer. Given the following chat messages, extract a weekly summary.\n"
        "Return ONLY valid JSON (no markdown) with this exact structure:\n"
        '{"key_decisions": ["...", "..."], "action_items": ["- [ ] Task @Owner by Date", ..
