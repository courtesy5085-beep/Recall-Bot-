import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict, Generator, List, Tuple

import chromadb
import streamlit as st
import tiktoken
from chromadb.config import Settings
from dateutil import parser as date_parser
from fastembed import TextEmbedding
from fpdf import FPDF
from openai import OpenAI
from streamlit_extras.stylable_container import stylable_container

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="RecallBot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "recallbot.db"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_MODEL = "gpt-4o-mini"

# =========================================================
# STYLES
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0A0A0B;
    color: #E5E7EB;
}

.stApp {
    background-color: #0A0A0B;
}

section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #1F2937;
}

div[data-testid="stChatMessage"] {
    background: #111111;
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 12px;
    animation: fadeIn 0.3s ease-in-out;
}

.stTextInput input,
.stSelectbox div,
.stTextArea textarea {
    background: #111111 !important;
    color: #E5E7EB !important;
    border: 1px solid #1F2937 !important;
    border-radius: 12px !important;
}

.stButton button {
    min-height: 44px;
    background: #6366F1;
    color: white;
    border: none;
    border-radius: 12px;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton button:hover {
    background: #7C82FF;
    transform: translateY(-1px);
}

.source-card {
    background: #111111;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
}

.code-font {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.shimmer {
    width: 100%;
    height: 8px;
    background: linear-gradient(90deg, #111111 25%, #6366F1 50%, #111111 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 999px;
    margin-bottom: 10px;
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

hr {
    border-color: #1F2937;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# DATABASE
# =========================================================


def init_sqlite() -> None:
    """Initialize SQLite tables."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            rating TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recent_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace TEXT,
            question TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()


init_sqlite()

# =========================================================
# CACHED RESOURCES
# =========================================================


@st.cache_resource
def get_chroma_client():
    """Return persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


@st.cache_resource
def get_embedder():
    """Return FastEmbed model."""
    return TextEmbedding(model_name=EMBED_MODEL)


# =========================================================
# OPENAI
# =========================================================


def get_openai_client() -> OpenAI | None:
    """Get OpenAI client."""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        return None


# =========================================================
# HELPERS
# =========================================================


def get_collection(workspace: str):
    """Get or create collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=workspace)


def tokenize_text(text: str) -> List[int]:
    """Tokenize text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)


def detokenize(tokens: List[int]) -> str:
    """Detokenize."""
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Chunk text into overlapping token windows."""
    tokens = tokenize_text(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = detokenize(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# =========================================================
# PARSERS
# =========================================================


def parse_whatsapp(content: str) -> List[Dict[str, Any]]:
    """Parse WhatsApp export."""
    pattern = r"\[(.*?)\]\s(.*?):\s(.*)"
    messages = []

    for line in content.splitlines():
        match = re.match(pattern, line)
        if match:
            dt, sender, message = match.groups()

            try:
                parsed_date = date_parser.parse(dt)
            except Exception:
                parsed_date = datetime.now()

            messages.append(
                {
                    "date": parsed_date.isoformat(),
                    "sender": sender.strip(),
                    "message": message.strip(),
                }
            )

    return messages


def parse_slack(content: str) -> List[Dict[str, Any]]:
    """Parse Slack JSON export."""
    data = json.loads(content)
    messages = []

    for item in data:
        text = item.get("text", "")
        user = item.get("user", "Unknown")
        ts = item.get("ts", "")

        try:
            dt = datetime.fromtimestamp(float(ts.split(".")[0]))
        except Exception:
            dt = datetime.now()

        messages.append(
            {
                "date": dt.isoformat(),
                "sender": user,
                "message": text,
            }
        )

    return messages


def auto_parse(file_name: str, content: str) -> List[Dict[str, Any]]:
    """Auto detect format."""
    if file_name.endswith(".json"):
        return parse_slack(content)
    return parse_whatsapp(content)


# =========================================================
# INDEXING
# =========================================================


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts."""
    embedder = get_embedder()
    embeddings = list(embedder.embed(texts))
    return [e.tolist() for e in embeddings]


def build_chunks(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create chunks."""
    chunks = []

    for msg in messages:
        combined = f"[{msg['date']}] {msg['sender']}: {msg['message']}"
        text_chunks = chunk_text(combined)

        for ch in text_chunks:
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": ch,
                    "date": msg["date"],
                    "sender": msg["sender"],
                    "message": msg["message"],
                }
            )

    return chunks


def index_messages(
    workspace: str,
    messages: List[Dict[str, Any]],
) -> Tuple[int, int, float]:
    """Index messages into ChromaDB."""
    collection = get_collection(workspace)

    start_time = time.time()
    chunks = build_chunks(messages)

    progress = st.empty()
    shimmer = st.empty()

    shimmer.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)

    batch_size = 128

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        progress.info(
            f"Indexing {len(messages):,} messages... "
            f"{min(i + batch_size, len(chunks)):,}/{len(chunks):,}"
        )

        embeddings = embed_texts([x["text"] for x in batch])

        collection.add(
            ids=[x["id"] for x in batch],
            documents=[x["text"] for x in batch],
            embeddings=embeddings,
            metadatas=[
                {
                    "date": x["date"],
                    "sender": x["sender"],
                    "message": x["message"],
                }
                for x in batch
            ],
        )

    shimmer.empty()
    progress.success(
        f"{len(messages):,} messages, "
        f"{len(chunks):,} chunks, indexed in "
        f"{round(time.time() - start_time, 1)}s"
    )

    return len(messages), len(chunks), round(time.time() - start_time, 1)


# =========================================================
# RAG
# =========================================================


def retrieve_context(
    workspace: str,
    query: str,
    top_k: int = 7,
):
    """Retrieve relevant chunks."""
    collection = get_collection(workspace)

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return docs, metas, distances


def build_prompt(query: str, docs: List[str], metas: List[Dict]) -> str:
    """Build RAG prompt."""
    context = []

    for doc, meta in zip(docs, metas):
        context.append(
            f"[{meta['date']}][{meta['sender']}]\n{doc}"
        )

    joined = "\n\n".join(context)

    return f"""
You are RecallBot.
Answer only using the [CONTEXT].
Cite sources as [Date][Sender].
If not in context, say you couldn't find it.

[CONTEXT]
{joined}

[QUESTION]
{query}
"""


def stream_openai_answer(prompt: str) -> Generator[str, None, None]:
    """Stream OpenAI answer."""
    client = get_openai_client()

    if not client:
        yield "OpenAI key missing."
        return

    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are RecallBot.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception:
        yield "OpenAI rate limit reached. Please try again in a moment."


# =========================================================
# SUMMARIZER
# =========================================================


def summarize_workspace(workspace: str) -> str:
    """Summarize last 7 days."""
    collection = get_collection(workspace)

    results = collection.get()

    docs = results.get("documents", [])[:120]

    if not docs:
        return "No data available."

    joined = "\n".join(docs)

    client = get_openai_client()

    if not client:
        return "Missing OpenAI key."

    prompt = f"""
Summarize these chat logs from the last 7 days.

Return markdown with:
- Key Decisions
- Action Items
- Unresolved Questions

Chat logs:
{joined}
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception:
        return "Unable to summarize."


# =========================================================
# PDF EXPORT
# =========================================================


def export_pdf(answer: str) -> bytes:
    """Export answer as PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    lines = answer.split("\n")

    for line in lines:
        pdf.multi_cell(0, 8, line)

    return bytes(pdf.output(dest="S").encode("latin-1"))


# =========================================================
# SQLITE HELPERS
# =========================================================


def save_feedback(question: str, answer: str, rating: str) -> None:
    """Save feedback."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO feedback (question, answer, rating)
        VALUES (?, ?, ?)
        """,
        (question, answer, rating),
    )

    conn.commit()
    conn.close()


def save_recent_question(workspace: str, question: str) -> None:
    """Save recent question."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO recent_questions (workspace, question)
        VALUES (?, ?)
        """,
        (workspace, question),
    )

    conn.commit()
    conn.close()


def get_recent_questions(workspace: str) -> List[str]:
    """Get recent questions."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT question
        FROM recent_questions
        WHERE workspace=?
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (workspace,),
    )

    rows = cur.fetchall()
    conn.close()

    return [x[0] for x in rows]


# =========================================================
# SESSION STATE
# =========================================================

if "workspace" not in st.session_state:
    st.session_state.workspace = "default"

if "messages" not in st.session_state:
    st.session_state.messages = {}

if st.session_state.workspace not in st.session_state.messages:
    st.session_state.messages[st.session_state.workspace] = []

if "retrieved_sources" not in st.session_state:
    st.session_state.retrieved_sources = []

# =========================================================
# LAYOUT
# =========================================================

left, center, right = st.columns([1.2, 2.5, 1.3])

# =========================================================
# SIDEBAR
# =========================================================

with left:
    st.subheader("RecallBot")

    workspace = st.text_input(
        "Workspace",
        value=st.session_state.workspace,
    )

    st.session_state.workspace = workspace

    if st.button("Switch Workspace"):
        st.rerun()

    if st.button("Delete Workspace"):
        client = get_chroma_client()
        try:
            client.delete_collection(workspace)
            st.success("Workspace deleted.")
        except Exception:
            st.warning("Workspace not found.")

    st.divider()

    uploaded = st.file_uploader(
        "Upload WhatsApp or Slack Export",
        type=["txt", "json"],
    )

    if uploaded:
        content = uploaded.read().decode("utf-8")
        parsed = auto_parse(uploaded.name, content)

        if st.button("Index Messages"):
            index_messages(workspace, parsed)

    if st.button("Use Sample Data"):
        sample = [
            {
                "date": datetime.now().isoformat(),
                "sender": "Sara",
                "message": "We should launch on Friday.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Alex",
                "message": "Need QA before deployment.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Sara",
                "message": "Marketing assets are ready.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "John",
                "message": "Client approved the final version.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Mina",
                "message": "Let's prepare release notes.",
            },
        ]

        index_messages(workspace, sample)

    st.divider()

    if st.button("Export Workspace"):
        collection = get_collection(workspace)
        data = collection.get()

        export_data = {
            "documents": data["documents"],
            "metadatas": data["metadatas"],
        }

        st.download_button(
            "Download JSON",
            json.dumps(export_data, indent=2),
            file_name=f"{workspace}.json",
        )

    st.divider()

    st.markdown("### Recent Questions")

    for q in get_recent_questions(workspace):
        if st.button(q[:40], key=q):
            st.session_state.prefill = q

# =========================================================
# CHAT CENTER
# =========================================================

with center:
    top1, top2 = st.columns([4, 1])

    with top1:
        st.subheader(f"Workspace: {workspace}")

    with top2:
        if st.button("Summarize"):
            summary = summarize_workspace(workspace)

            with st.chat_message("assistant"):
                st.markdown(summary)

    for msg in st.session_state.messages[workspace]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("👍", key=f"up_{msg['id']}"):
                        save_feedback(
                            msg.get("question", ""),
                            msg["content"],
                            "up",
                        )

                with col2:
                    if st.button("👎", key=f"down_{msg['id']}"):
                        save_feedback(
                            msg.get("question", ""),
                            msg["content"],
                            "down",
                        )

    query = st.chat_input(
        "Ask RecallBot...",
    )

    if not query and "prefill" in st.session_state:
        query = st.session_state.prefill
        del st.session_state.prefill

    if query:
        save_recent_question(workspace, query)

        st.session_state.messages[workspace].append(
            {
                "role": "user",
                "content": query,
                "id": str(uuid.uuid4()),
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        docs, metas, distances = retrieve_context(workspace, query)

        st.session_state.retrieved_sources = list(zip(docs, metas))

        if distances and distances[0] > 0.7:
            answer = "I couldn’t find that in the chat history."

            with st.chat_message("assistant"):
                st.markdown(answer)

        else:
            prompt = build_prompt(query, docs, metas)

            with st.chat_message("assistant"):
                response = st.write_stream(
                    stream_openai_answer(prompt)
                )

                source_lines = []

                for meta in metas:
                    dt = date_parser.parse(meta["date"])

 
