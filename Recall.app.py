import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from datetime import datetime
from dateutil import parser
from fpdf import FPDF
import sqlite3
import json
import uuid
import re
import time
import tiktoken

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="RecallBot",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CSS
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    background: #0A0A0B;
    color: #E5E7EB;
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0A0A0B;
}

section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #1F2937;
}

.stButton button {
    background: #6366F1;
    color: white;
    border-radius: 12px;
    border: none;
    min-height: 44px;
    width: 100%;
}

.stButton button:hover {
    background: #7C82FF;
}

.stChatMessage {
    background: #111111;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 10px;
}

div[data-testid="stChatMessage"] {
    background: #111111;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}

.shimmer {
  width: 100%;
  height: 8px;
  background: linear-gradient(90deg,#111111 25%,#6366F1 50%,#111111 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 999px;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
""", unsafe_allow_html=True)

# =========================
# DATABASE
# =========================

conn = sqlite3.connect("recallbot.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS feedback(
id INTEGER PRIMARY KEY AUTOINCREMENT,
question TEXT,
answer TEXT,
rating TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS recent_questions(
id INTEGER PRIMARY KEY AUTOINCREMENT,
workspace TEXT,
question TEXT
)
""")

conn.commit()

# =========================
# OPENAI
# =========================

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except:
    st.error("Add OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

# =========================
# CHROMA
# =========================

@st.cache_resource
def load_chroma():
    return chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

chroma_client = load_chroma()
embed_model = load_embed_model()

# =========================
# SESSION STATE
# =========================

if "workspace" not in st.session_state:
    st.session_state.workspace = "default"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if st.session_state.workspace not in st.session_state.chat_history:
    st.session_state.chat_history[st.session_state.workspace] = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# =========================
# HELPERS
# =========================

def get_collection(name):
    return chroma_client.get_or_create_collection(name=name)

def chunk_text(text, chunk_size=500, overlap=100):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def embed_texts(texts):
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True
    )
    return embeddings.tolist()

# =========================
# PARSERS
# =========================

def parse_whatsapp(content):
    pattern = r"\[(.*?)\]\s(.*?):\s(.*)"
    messages = []

    for line in content.splitlines():
        match = re.match(pattern, line)

        if match:
            dt, sender, message = match.groups()

            try:
                parsed = parser.parse(dt)
            except:
                parsed = datetime.now()

            messages.append({
                "date": parsed.isoformat(),
                "sender": sender,
                "message": message
            })

    return messages

def parse_slack(content):
    data = json.loads(content)

    messages = []

    for item in data:
        ts = item.get("ts", "")
        user = item.get("user", "Unknown")
        text = item.get("text", "")

        try:
            dt = datetime.fromtimestamp(float(ts.split(".")[0]))
        except:
            dt = datetime.now()

        messages.append({
            "date": dt.isoformat(),
            "sender": user,
            "message": text
        })

    return messages

def auto_parse(name, content):
    if name.endswith(".json"):
        return parse_slack(content)
    return parse_whatsapp(content)

# =========================
# INDEXING
# =========================

def index_messages(workspace, messages):

    collection = get_collection(workspace)

    all_chunks = []

    for msg in messages:

        full_text = f"""
[{msg['date']}]
{msg['sender']}:
{msg['message']}
"""

        chunks = chunk_text(full_text)

        for ch in chunks:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": ch,
                "sender": msg["sender"],
                "date": msg["date"],
                "message": msg["message"]
            })

    progress = st.empty()
    shimmer = st.empty()

    shimmer.markdown(
        '<div class="shimmer"></div>',
        unsafe_allow_html=True
    )

    start = time.time()

    batch_size = 128

    for i in range(0, len(all_chunks), batch_size):

        batch = all_chunks[i:i+batch_size]

        progress.info(
            f"Indexing {len(messages):,} messages... "
            f"{min(i+batch_size, len(all_chunks)):,}/{len(all_chunks):,}"
        )

        embeddings = embed_texts(
            [x["text"] for x in batch]
        )

        collection.add(
            ids=[x["id"] for x in batch],
            documents=[x["text"] for x in batch],
            embeddings=embeddings,
            metadatas=[
                {
                    "sender": x["sender"],
                    "date": x["date"],
                    "message": x["message"]
                }
                for x in batch
            ]
        )

    shimmer.empty()

    progress.success(
        f"{len(messages):,} messages, "
        f"{len(all_chunks):,} chunks, "
        f"indexed in {round(time.time()-start,1)}s"
    )

# =========================
# RAG
# =========================

def retrieve_context(workspace, query):

    collection = get_collection(workspace)

    q_embed = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=7
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return docs, metas, distances

def build_prompt(query, docs, metas):

    context = []

    for d, m in zip(docs, metas):

        context.append(
            f"""
[{m['date']}][{m['sender']}]
{d}
"""
        )

    context_text = "\n".join(context)

    return f"""
You are RecallBot.

Answer ONLY using the provided context.

If answer is missing, say:
"I couldn’t find that in the chat history."

Always cite:
[Date][Sender]

[CONTEXT]
{context_text}

[QUESTION]
{query}
"""

# =========================
# PDF
# =========================

def export_pdf(text):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    return bytes(
        pdf.output(dest="S").encode("latin-1")
    )

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("RecallBot")

    workspace = st.text_input(
        "Workspace",
        value=st.session_state.workspace
    )

    st.session_state.workspace = workspace

    if st.button("Switch Workspace"):
        st.rerun()

    if st.button("Delete Workspace"):
        try:
            chroma_client.delete_collection(workspace)
            st.success("Workspace deleted.")
        except:
            st.warning("Workspace not found.")

    st.divider()

    uploaded = st.file_uploader(
        "Upload WhatsApp / Slack",
        type=["txt", "json"]
    )

    if uploaded:

        content = uploaded.read().decode("utf-8")

        parsed = auto_parse(
            uploaded.name,
            content
        )

        if st.button("Index Messages"):
            index_messages(
                workspace,
                parsed
            )

    if st.button("Use Sample Data"):

        sample = [
            {
                "date": datetime.now().isoformat(),
                "sender": "Sara",
                "message": "We launch Friday."
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Ali",
                "message": "QA testing is complete."
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "John",
                "message": "Client approved final build."
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Mina",
                "message": "Marketing assets ready."
            },
            {
                "date": datetime.now().isoformat(),
                "sender": "Sara",
                "message": "Push update tonight."
            }
        ]

        index_messages(workspace, sample)

    st.divider()

    if st.button("Export Workspace"):

        collection = get_collection(workspace)

        data = collection.get()

        export_data = {
            "documents": data["documents"],
            "metadatas": data["metadatas"]
        }

        st.download_button(
            "Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"{workspace}.json"
        )

    st.divider()

    st.subheader("Recent Questions")

    cur.execute("""
    SELECT question
    FROM recent_questions
    WHERE workspace=?
    ORDER BY id DESC
    LIMIT 10
    """, (workspace,))

    recent = cur.fetchall()

    for q in recent:

        if st.button(q[0][:40], key=q[0]):
            st.session_state.prefill = q[0]

# =========================
# MAIN UI
# =========================

col1, col2 = st.columns([3,1])

with col1:

    top1, top2 = st.columns([4,1])

    with top1:
        st.title("RecallBot")

    with top2:

        if st.button("Summarize"):

            collection = get_collection(workspace)

            data = collection.get()

            docs = data["documents"][:120]

            prompt = f"""
Summarize chats into:

1. Key Decisions
2. Action Items
3. Unresolved Questions

Chats:
{docs}
"""

            try:

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                st.markdown(
                    response.choices[0].message.content
                )

            except:
                st.error("Summary failed.")

    history = st.session_state.chat_history[
        workspace
    ]

    for msg in history:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg["role"] == "assistant":

                c1, c2 = st.columns(2)

                with c1:
                    if st.button(
                        "👍",
                        key=f"up_{msg['id']}"
                    ):

                        cur.execute("""
                        INSERT INTO feedback(
                        question,
                        answer,
                        rating
                        )
                        VALUES(?,?,?)
                        """, (
                            msg.get("question",""),
                            msg["content"],
                            "up"
                        ))

                        conn.commit()

                with c2:
                    if st.button(
                        "👎",
                        key=f"down_{msg['id']}"
                    ):

                        cur.execute("""
                        INSERT INTO feedback(
                        question,
                        answer,
                        rating
                        )
                        VALUES(?,?,?)
                        """, (
                            msg.get("question",""),
                            msg["content"],
                            "down"
                        ))

                        conn.commit()

    prompt = st.chat_input(
        "Ask RecallBot..."
    )

    if not prompt and "prefill" in st.session_state:
        prompt = st.session_state.prefill
        del st.session_state.prefill

    if prompt:

        cur.execute("""
        INSERT INTO recent_questions(
        workspace,
        question
        )
        VALUES(?,?)
        """, (
            workspace,
            prompt
        ))

        conn.commit()

        history.append({
            "role": "user",
            "content": prompt,
            "id": str(uuid.uuid4())
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        docs, metas, distances = retrieve_context(
            workspace,
            prompt
        )

        st.session_state.sources = list(
            zip(docs, metas)
        )

        if distances and distances[0] > 0.7:

            answer = (
                "I couldn’t find that "
                "in the chat history."
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

        else:

            rag_prompt = build_prompt(
                prompt,
                docs,
                metas
            )

            with st.chat_message("assistant"):

                try:

                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": rag_prompt
                            }
                        ],
                        stream=True
                    )

                    full_answer = ""

                    placeholder = st.empty()

                    for chunk in stream:

                        delta = (
                            chunk.choices[0]
                            .delta.content
                        )

                        if delta:
                            full_answer += delta
                            placeholder.markdown(
                                full_answer
                            )

                    citations = []

                    for m in metas:

                        dt = parser.parse(m["date"])

                        citations.append(
                            f"[{dt.strftime('%d %b')}, {m['sender']}]"
                        )

                    full_answer += (
                        "\n\n**Sources:** "
                        + ", ".join(citations)
                    )

                    placeholder.markdown(
                        full_answer
                    )

                    history.append({
                        "role": "assistant",
                        "content": full_answer,
                        "id": str(uuid.uuid4()),
                        "question": prompt
                    })

                    pdf_data = export_pdf(
                        full_answer
                    )

                    st.download_button(
                        "Export PDF",
                        data=pdf_data,
                        file_name="answer.pdf",
                        mime="application/pdf"
                    )

                except:
                    st.error(
                        "OpenAI rate limit reached."
                    )

with col2:

    st.subheader("Sources")

    if st.session_state.sources:

        for i, (doc, meta) in enumerate(
            st.session_state.sources
        ):

            with st.container(border=True):

                st.markdown(
                    f"""
**{meta['sender']}**

{doc[:300]}
"""
                )

                with st.expander(
                    "View Full Message"
                ):

                    st.markdown(
                        meta["message"]
                    )
