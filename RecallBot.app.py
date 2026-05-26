import streamlit as st
import re
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import tiktoken
import chromadb
from fastembed import TextEmbedding
from openai import OpenAI
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.icons import icon
from fpdf import FPDF

st.set_page_config(page_title="RecallBot", layout="wide", initial_sidebar_state="expanded")

# ================== CSS & STYLING ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.stApp {background: #0A0A0B; color: #E5E7EB;}
.stButton button {
    border-radius: 8px; border: 1px solid #1F2937; background: #111113;
    min-height: 44px; transition: all 0.2s ease; color: #E5E7EB;
}
.stButton button:hover {border-color: #6366F1; background: #1A1A1D;}
.stChatMessage {
    background: #111113; border: 1px solid #1F2937; border-radius: 12px;
    padding: 16px; animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(4px);}
    to {opacity: 1; transform: translateY(0);}
}
.progress-shimmer {
    background: linear-gradient(90deg, #1F2937 25%, #6366F1 50%, #1F2937 75%);
    background-size: 200% 100%; animation: shimmer 1.5s infinite;
    height: 6px; border-radius: 4px;
}
@keyframes shimmer {0% {background-position: 200% 0;} 100% {background-position: -200% 0;}}
.highlight {background: #6366F1; color: white; padding: 2px 4px; border-radius: 4px;}
@media (max-width: 768px) {
  .stColumns {flex-direction: column;}
}
code, pre {font-family: 'JetBrains Mono', monospace;}
</style>
""", unsafe_allow_html=True)

# ================== CONFIG ==================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 7
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
CHROMA_PATH = "./chroma_db"
SIMILARITY_THRESHOLD = 0.3

# ================== INIT ==================
@st.cache_resource
def get_client():
    key = st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets")
        st.stop()
    return OpenAI(api_key=key)

@st.cache_resource
def get_embedder():
    return TextEmbedding(model_name=EMBED_MODEL)

@st.cache_resource
def get_chroma():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def init_db():
    conn = sqlite3.connect("recallbot.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS feedback
                 (id TEXT PRIMARY KEY, workspace TEXT, rating INT, timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS recent_questions
                 (id TEXT PRIMARY KEY, workspace TEXT, question TEXT, timestamp TEXT)""")
    conn.commit()
    return conn

conn = init_db()

# ================== PARSERS & UTILS ==================
def parse_whatsapp(text: str) -> list:
    """Parse WhatsApp txt export. Format: dd/mm/yy, hh:mm - Sender: Message"""
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)"
    messages = []
    for line in text.split("\n"):
        match = re.match(pattern, line)
        if match:
            date, time, sender, msg = match.groups()
            try:
                dt = date_parser.parse(f"{date} {time}", dayfirst=True)
                messages.append({"date": dt.isoformat(), "sender": sender.strip(), "message": msg.strip()})
            except:
                continue
    return messages

def parse_slack(data: list) -> list:
    """Parse Slack JSON export"""
    messages = []
    for channel in data:
        for msg in channel.get("messages", []):
            if "text" in msg and "user" in msg:
                try:
                    ts = float(msg["ts"])
                    messages.append({
                        "date": datetime.fromtimestamp(ts).isoformat(),
                        "sender": msg["user"],
                        "message": msg["text"]
                    })
                except:
                    continue
    return messages

def chunk_messages(messages: list, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    """Chunk messages using tiktoken with overlap"""
    enc = tiktoken.get_encoding("cl100k_base")
    chunks, current_chunk, current_tokens = [], 0
    for msg in messages:
        text = f"[{msg['date'][:10]}, {msg['sender']}] {msg['message']}"
        tokens = len(enc.encode(text))
        if current_tokens + tokens > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else []
            current_tokens = sum(len(enc.encode(t)) for t in current_chunk)
        current_chunk.append(text)
        current_tokens += tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_collection(workspace_id: str):
    client = get_chroma()
    return client.get_or_create_collection(workspace_id)

def add_to_collection(workspace_id: str, chunks: list, metadatas: list) -> int:
    collection = get_collection(workspace_id)
    embedder = get_embedder()
    embeddings = list(embedder.embed(chunks, batch_size=128))
    ids = [hashlib.md5(c.encode()).hexdigest() for c in chunks]
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return len(chunks)

def query_collection(workspace_id: str, query: str, k=RETRIEVAL_K):
    collection = get_collection(workspace_id)
    embedder = get_embedder()
    query_emb = list(embedder.embed([query]))[0]
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    return results

def highlight_text(text: str, query: str) -> str:
    for word in query.split():
        if len(word) > 2:
            text = re.sub(f"({re.escape(word)})", r'<span class="highlight">\1</span>', text, flags=re.I)
    return text

# ================== SESSION STATE ==================
if "workspace" not in st.session_state:
    st.session_state.workspace = "default"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "source_docs" not in st.session_state:
    st.session_state.source_docs = []
if "selected_source_idx" not in st.session_state:
    st.session_state.selected_source_idx = None

# ================== SIDEBAR ==================
with st.sidebar:
    icon("database", size=20)
    st.markdown("### Workspaces")
    workspace = st.text_input("Workspace ID", value=st.session_state.workspace)
    if workspace!= st.session_state.workspace:
        st.session_state.workspace = workspace
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("### Upload")
    uploaded = st.file_uploader("Upload chat export", type=["txt", "json"])

    if st.button("Use Sample Data", use_container_width=True):
        sample = """[12/10/2024, 10:00] Alice: Let's set pricing at $29/month
[12/10/2024, 10:01] Bob: Agreed. Deadline is 15th Dec
[12/10/2024, 10:02] Alice: Who will handle marketing?
[12/10/2024, 10:03] Bob: I'll take care of it
[12/10/2024, 10:04] Alice: Perfect"""
        messages = parse_whatsapp(sample)
        chunks = chunk_messages(messages)
        add_to_collection(workspace, chunks, messages)
        st.success(f"Indexed {len(messages)} messages")

    if uploaded:
        content = uploaded.read()
        start_time = time.time()
        try:
            if uploaded.name.endswith(".json"):
                messages = parse_slack(json.loads(content))
            else:
                messages = parse_whatsapp(content.decode("utf-8"))

            if not messages:
                st.error("No messages parsed. Check file format.")
            else:
                progress_bar = st.progress(0)
                status = st.empty()
                total = len(messages)
                chunks = chunk_messages(messages)

                for i in range(0, len(chunks), 10):
                    batch = chunks[i:i+10]
                    batch_meta = messages[i:i+10] if i < len(messages) else messages[-len(batch):]
                    add_to_collection(workspace, batch, batch_meta)
                    progress = min((i+10)/len(chunks), 1.0)
                    progress_bar.progress(progress)
                    status.markdown(f'<div class="progress-shimmer"></div>', unsafe_allow_html=True)
                    status.text(f"Indexing {total} messages... {min(i+10, total)}/{total}")
                    time.sleep(0.05)

                elapsed = round(time.time() - start_time, 1)
                status.success(f"{total} messages, {len(chunks)} chunks indexed in {elapsed}s")
        except Exception as e:
            st.error(f"Parse error: {e}")

    st.markdown("### Recent Questions")
    c = conn.cursor()
    c.execute("SELECT question FROM recent_questions WHERE workspace=? ORDER BY timestamp DESC LIMIT 10", (workspace,))
    for row in c.fetchall():
        if st.button(row[0], key=f"recent_{row[0]}", use_container_width=True):
            st.session_state.pending_query = row[0]

    st.markdown("### Export")
    if st.button("Export Workspace", use_container_width=True):
        col = get_collection(workspace)
        data = col.get()
        st.download_button("Download JSON", json.dumps(data, indent=2),
                          file_name=f"{workspace}.json", mime="application/json")

# ================== MAIN LAYOUT ==================
left, right = st.columns([0.7, 0.3])

with left:
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown("### RecallBot")
    with col2:
        if st.button("Summarize", use_container_width=True):
            st.session_state.summarize = True

    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about your chats..."):
        st.session_state.pending_query = prompt

    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query

        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        try:
            results = query_collection(workspace, query)

            if not results["documents"][0] or max(results["distances"][0]) > 1 - SIMILARITY_THRESHOLD:
                response = "I couldn’t find that in the chat history."
                st.session_state.source_docs = []
            else:
                context = "\n\n".join(results["documents"][0])
                system_prompt = f"""You are RecallBot, an AI memory assistant.
Answer using ONLY the [CONTEXT] provided.
If answer isn't in context, say: "I couldn’t find that in the chat history."
Cite sources as [Date, Sender].
[CONTEXT]
{context}
[/CONTEXT]"""

                client = get_client()
                stream = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "system", "content": system_prompt},
                             {"role": "user", "content": query}],
                    stream=True,
                    temperature=0.2
                )

                with st.chat_message("assistant"):
                    response = st.write_stream(stream)

                st.session_state.source_docs = results

            st.session_state.chat_history.append({"role": "assistant", "content": response})

            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO recent_questions VALUES (?,?,?,?)",
                      (hashlib.md5(query.encode()).hexdigest(), workspace, query, datetime.now().isoformat()))
            conn.commit()

        except Exception as e:
            st.error(f"Error: {str(e)}. Please try again.")

    if st.session_state.get("summarize"):
        del st.session_state.summarize
        results = query_collection(workspace, "key decisions action items unresolved questions", k=15)
        if results["documents"][0]:
            context = "\n\n".join(results["documents"][0])
            prompt = f"""Summarize this chat into JSON with keys: key_decisions, action_items, unresolved_questions.
Context: {context}"""
            try:
                client = get_client()
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                data = json.loads(resp.choices[0].message.content)

                md = "### Key Decisions\n"
                md += "\n".join(f"- {d}" for d in data.get("key_decisions", []))
                md += "\n\n### Action Items\n"
                md += "\n".join(f"- [ ] {a}" for a in data.get("action_items", []))
                md += "\n\n### Unresolved Questions\n"
                md += "\n".join(f"Q: {q}" for q in data.get("unresolved_questions", []))
                st.markdown(md)
            except Exception as e:
                st.error(f"Summarization failed: {e}")

    if st.session_state.chat_history:
        if st.button("Export Answer as PDF"):
            last_msg = st.session_state.chat_history[-1]["content"]
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, last_msg[:2000])
            pdf.output("/tmp/answer.pdf")
            with open("/tmp/answer.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name="answer.pdf", mime="application/pdf")

# ================== SOURCE PANEL ==================
with right:
    icon("file-text", size=20)
    st.markdown("### Sources")
    if st.session_state.source_docs and st.session_state.source_docs["documents"][0]:
        query = st.session_state.chat_history[-2]["content"] if len(st.session_state.chat_history) > 1 else ""
        for i, doc in enumerate(st.session_state.source_docs["documents"][0]):
            with stylable_container(key=f"src{i}", css_styles="border:1px solid #1F2937; padding:10px; border-radius:8px; margin-bottom:8px; cursor:pointer;"):
                highlighted = highlight_text(doc[:200] + "...", query)
                st.markdown(highlighted, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("View", key=f"view{i}"):
                        st.session_state.selected_source_idx = i
                with col2:
                    if st.button("👍", key=f"up{i}"):
                        c = conn.cursor()
                        c.execute("INSERT INTO feedback VALUES (?,?,?,?)",
                                  (f"{workspace}_{i}", workspace, 1, datetime.now().isoformat()))
                        conn.commit()
                with col3:
                    if st.button("👎", key=f"down{i}"):
                        c = conn.cursor()
                        c.execute("INSERT INTO feedback VALUES (?,?,?,?)",
                                  (f"{workspace}_{i}", workspace, 0, datetime.now().isoformat()))
                        conn.commit()

        if st.session_state.selected_source_idx is not None:
            idx = st.session_state.selected_source_idx
            st.markdown("---")
            st.markdown("### Full Context")
            st.text_area("Message",
                        st.session_state.source_docs["documents"][0][idx],
                        height=200, disabled=True)
    else:
        st.markdown("Ask a question to see sources")
