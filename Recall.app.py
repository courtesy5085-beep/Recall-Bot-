import os
import re
import json
import time
import uuid
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator

import orjson
import tiktoken
import streamlit as st
from dateutil import parser as date_parser
from fastembed import TextEmbedding
from fpdf import FPDF
from openai import OpenAI
from streamlit_extras.stylable_container import stylable_container

import chromadb
from chromadb.config import Settings


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RecallBot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DARK LINEAR STYLE CSS
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0A0A0B;
    --card: #111113;
    --accent: #6366F1;
    --text: #E5E7EB;
    --muted: #9CA3AF;
    --border: #1F2937;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

section[data-testid="stSidebar"] {
    background: #0F0F10;
    border-right: 1px solid var(--border);
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

[data-testid="stChatMessage"] {
    background: #111113;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
    animation: fadeIn 0.3s ease-in-out;
}

.stButton > button {
    width: 100%;
    min-height: 44px;
    border-radius: 12px;
    background: #151517;
    border: 1px solid var(--border);
    color: var(--text);
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: var(--accent);
    background: #1A1A1F;
}

.stTextInput > div > div > input {
    background: #111113;
    color: white;
    border-radius: 12px;
    border: 1px solid var(--border);
}

.stTextArea textarea {
    background: #111113;
    color: white;
    border-radius: 12px;
    border: 1px solid var(--border);
}

.stFileUploader {
    border-radius: 16px;
    border: 1px dashed var(--border);
    background: #111113;
    padding: 16px;
}

.source-card {
    background: #111113;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}

.code-font {
    font-family: 'JetBrains Mono', monospace;
}

.shimmer {
    width: 100%;
    height: 8px;
    background: linear-gradient(
        90deg,
        #1F2937 25%,
        #6366F1 50%,
        #1F2937 75%
    );
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 999px;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

hr {
    border-color: var(--border);
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# DATABASE
# =========================================================

DB_PATH = "recallbot.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

cur.execute(
    """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT,
    feedback TEXT,
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

# =========================================================
# OPENAI
# =========================================================

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# CACHE
# =========================================================


@st.cache_resource
def get_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )


embedding_model = get_embedding_model()
chroma_client = get_chroma_client()

# =========================================================
# HELPERS
# =========================================================


def generate_embedding(texts: List[str]) -> List[List[float]]:
    embeddings = list(embedding_model.embed(texts))
    return [e.tolist() for e in embeddings]


def tokenize_text(text: str) -> List[int]:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)


def detokenize(tokens: List[int]) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    tokens = tokenize_text(text)
    chunks = []

    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(detokenize(chunk))
        start += chunk_size - overlap

    return chunks


def parse_whatsapp(content: str) -> List[Dict[str, Any]]:
    pattern = r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\]\s([^:]+):\s(.+)$"

    messages = []

    for line in content.splitlines():
        match = re.match(pattern, line)

        if match:
            date_str, time_str, sender, message = match.groups()

            try:
                dt = datetime.strptime(
                    f"{date_str} {time_str}",
                    "%d/%m/%y %H:%M"
                )
            except:
                dt = datetime.now()

            messages.append(
                {
                    "date": dt.isoformat(),
                    "sender": sender.strip(),
                    "message": message.strip(),
                }
            )

    return messages


def parse_slack(content: str) -> List[Dict[str, Any]]:
    data = orjson.loads(content)

    messages = []

    for item in data:
        text = item.get("text", "")
        user = item.get("user", "Unknown")
        ts = item.get("ts", "")

        try:
            dt = datetime.fromtimestamp(float(ts))
        except:
            dt = datetime.now()

        messages.append(
            {
                "date": dt.isoformat(),
                "sender": user,
                "message": text,
            }
        )

    return messages


def parse_uploaded_file(uploaded_file) -> List[Dict[str, Any]]:
    content = uploaded_file.read().decode("utf-8", errors="ignore")

    if uploaded_file.name.endswith(".json"):
        return parse_slack(content)

    return parse_whatsapp(content)


def create_workspace(workspace: str):
    try:
        chroma_client.get_collection(workspace)
    except:
        chroma_client.create_collection(
            name=workspace,
            metadata={"hnsw:space": "cosine"},
        )


def delete_workspace(workspace: str):
    try:
        chroma_client.delete_collection(workspace)
    except:
        pass


def get_collection(workspace: str):
    return chroma_client.get_collection(workspace)


def index_messages(workspace: str, messages: List[Dict[str, Any]]):
    collection = get_collection(workspace)

    docs = []
    metas = []
    ids = []

    for msg in messages:
        text = f"[{msg['date']}][{msg['sender']}] {msg['message']}"

        chunks = chunk_text(text)

        for chunk in chunks:
            docs.append(chunk)

            metas.append(
                {
                    "sender": msg["sender"],
                    "date": msg["date"],
                }
            )

            ids.append(str(uuid.uuid4()))

    total = len(docs)

    shimmer = st.empty()
    progress_text = st.empty()

    shimmer.markdown(
        "<div class='shimmer'></div>",
        unsafe_allow_html=True,
    )

    batch_size = 128

    start_time = time.time()

    for i in range(0, total, batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_meta = metas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        embeddings = generate_embedding(batch_docs)

        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
            ids=batch_ids,
        )

        progress_text.markdown(
            f"""
            <div class='code-font'>
            Indexing {total:,} chunks... {min(i + batch_size, total):,}/{total:,}
            </div>
            """,
            unsafe_allow_html=True,
        )

    duration = round(time.time() - start_time, 1)

    shimmer.empty()

    progress_text.success(
        f"{len(messages):,} messages, {total:,} chunks indexed in {duration}s"
    )


def retrieve_context(
    workspace: str,
    query: str,
    top_k: int = 7,
):
    collection = get_collection(workspace)

    q_emb = generate_embedding([query])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )

    return results


def build_context(results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []

    for doc, meta in zip(docs, metas):
        context_parts.append(
            f"[{meta['date']}][{meta['sender']}]\n{doc}"
        )

    return "\n\n".join(context_parts)


def stream_openai_response(prompt: str) -> Generator[str, None, None]:
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are RecallBot. "
                        "Answer only using the [CONTEXT]. "
                        "If not in context say you couldn't find it. "
                        "Cite sources as [Date][Sender]."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta

    except Exception:
        yield "OpenAI rate limit reached. Please wait and try again."


def save_recent_question(workspace: str, question: str):
    cur.execute(
        """
        INSERT INTO recent_questions (workspace, question)
        VALUES (?, ?)
        """,
        (workspace, question),
    )
    conn.commit()


def save_feedback(question: str, answer: str, feedback: str):
    cur.execute(
        """
        INSERT INTO feedback (question, answer, feedback)
        VALUES (?, ?, ?)
        """,
        (question, answer, feedback),
    )
    conn.commit()


def export_workspace(workspace: str):
    collection = get_collection(workspace)

    data = collection.get()

    return json.dumps(data, indent=2)


def create_pdf(text: str):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_auto_page_break(True, margin=15)

    pdf.set_font("Helvetica", size=12)

    pdf.multi_cell(0, 8, text)

    path = "recallbot_export.pdf"

    pdf.output(path)

    return path


def summarize_workspace(workspace: str):
    collection = get_collection(workspace)

    data = collection.get()

    docs = data.get("documents", [])[:50]

    joined = "\n".join(docs)

    prompt = f"""
Summarize the last 7 days.

Return markdown with:
- Key Decisions
- Action Items
- Unresolved Questions

Context:
{joined}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content


# =========================================================
# SESSION STATE
# =========================================================

if "workspace" not in st.session_state:
    st.session_state.workspace = "default"

if "messages" not in st.session_state:
    st.session_state.messages = {}

if st.session_state.workspace not in st.session_state.messages:
    st.session_state.messages[st.session_state.workspace] = []

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## RecallBot")

    workspace = st.text_input(
        "Workspace",
        value=st.session_state.workspace,
    )

    if workspace:
        st.session_state.workspace = workspace
        create_workspace(workspace)

    if st.button("Create / Switch Workspace"):
        create_workspace(workspace)
        st.success(f"Workspace: {workspace}")

    if st.button("Delete Workspace"):
        delete_workspace(workspace)
        st.warning("Workspace deleted.")

    uploaded = st.file_uploader(
        "Upload WhatsApp / Slack",
        type=["txt", "json"],
    )

    if uploaded:
        msgs = parse_uploaded_file(uploaded)

        index_messages(workspace, msgs)

    if st.button("Use Sample Data"):
        sample = [
            {
                "date": datetime.now().isoformat(),
                "sender": "Sara",
                "message": "We should launch RecallBot next Friday.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Ali",
                "message": "Design team approved the dark Linear UI.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "John",
                "message": "Need export PDF feature before launch.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Sara",
                "message": "Let's add citations to every AI response.",
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Ali",
                "message": "Workspace isolation is now complete.",
            },
        ]

        index_messages(workspace, sample)

    st.divider()

    st.markdown("### Recent Questions")

    rows = cur.execute(
        """
        SELECT question
        FROM recent_questions
        WHERE workspace=?
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (workspace,),
    ).fetchall()

    for row in rows:
        if st.button(row[0], key=row[0]):
            st.session_state.prefill = row[0]

    st.divider()

    if st.button("Export Workspace"):
        data = export_workspace(workspace)

        st.download_button(
            "Download JSON",
            data=data,
            file_name=f"{workspace}.json",
            mime="application/json",
        )

# =========================================================
# MAIN LAYOUT
# =========================================================

left, center, right = st.columns([1.1, 2.6, 1.2])

# =========================================================
# CENTER CHAT
# =========================================================

with center:

    top1, top2 = st.columns([4, 1])

    with top1:
        st.markdown(f"## {workspace}")

    with top2:
        if st.button("Summarize"):
            summary = summarize_workspace(workspace)

            st.markdown(summary)

    chat_history = st.session_state.messages.setdefault(
        workspace,
        []
    )

    for idx, msg in enumerate(chat_history):

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":

                col1, col2, col3 = st.columns([1, 1, 4])

                with col1:
                    if st.button("👍", key=f"up_{idx}"):
                        save_feedback(
                            "question",
                            msg["content"],
                            "up",
                        )

                with col2:
                    if st.button("👎", key=f"down_{idx}"):
                        save_feedback(
                            "question",
                            msg["content"],
                            "down",
                        )

                with col3:
                    if st.button("Export PDF", key=f"pdf_{idx}"):

                        pdf_path = create_pdf(msg["content"])

                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "Download PDF",
                                f,
                                file_name="answer.pdf",
                            )

    user_query = st.chat_input(
        "Ask RecallBot anything..."
    )

    if user_query:

        save_recent_question(workspace, user_query)

        chat_history.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        results = retrieve_context(
            workspace,
            user_query,
        )

        docs = results["documents"][0]

        if not docs:
            with st.chat_message("assistant"):
                st.markdown(
                    "I couldn’t find that in the chat history."
                )

        else:

            context = build_context(results)

            prompt = f"""
[CONTEXT]
{context}

[QUESTION]
{user_query}
"""

            with st.chat_message("assistant"):

                response = st.write_stream(
                    stream_openai_response(prompt)
                )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            st.session_state.latest_results = results

# =========================================================
# RIGHT SOURCES PANEL
# =========================================================

with right:

    st.markdown("### Sources")

    results = st.session_state.get("latest_results")

    if results:

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for i, (doc, meta) in enumerate(zip(docs, metas)):

            with stylable_container(
                key=f"source_{i}",
                css_styles="""
                {
                    border: 1px solid #1F2937;
                    border-radius: 14px;
                    padding: 12px;
                    background: #111113;
                    margin-bottom: 10px;
                }
                """,
            ):

                st.markdown(
                    f"""
                    <div class='code-font'>
                    [{meta['sender']}]<br>
                    {meta['date'][:10]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                short_doc = doc[:220]

                st.markdown(short_doc)

                with st.expander("View Context"):

                    st.markdown(doc)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<hr>
<div style='text-align:center;color:#9CA3AF;font-size:13px'>
RecallBot • AI Memory for Team Chats
</div>
""",
    unsafe_allow_html=True,
)
