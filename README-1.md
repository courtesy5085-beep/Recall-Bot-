# 🧠 RecallBot — AI Memory Assistant for Team Chats

> Upload your WhatsApp or Slack exports and ask AI questions about your chat history — with exact cited source messages.

![RecallBot Banner](https://img.shields.io/badge/RecallBot-AI%20Memory-6c63ff?style=for-the-badge&logo=openai&logoColor=white)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 💬 **AI Chat** | Ask natural language questions, get GPT-4o-mini answers with cited messages |
| 📤 **Import Chats** | WhatsApp `.txt` exports and Slack `.json`/`.zip` exports |
| 🔍 **Semantic Search** | ChromaDB-powered vector search across all messages |
| 📊 **Analytics** | Message counts, timelines, heatmaps, keyword clouds |
| 📌 **Memory** | Save Q&As, bookmarks, AI-generated summaries |
| 🎨 **Dark/Light Mode** | Premium glassmorphism UI with gradient design |
| 🔒 **Auth System** | Secure login/signup with hashed passwords |

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/yourname/recallbot.git
cd recallbot
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
```

Or set it directly in the app via **Settings → OpenAI API Key**.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial RecallBot commit"
git branch -M main
git remote add origin https://github.com/yourname/recallbot.git
git push -u origin main
```

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **Deploy**

### 3. Add secrets in Streamlit Cloud

In your app's **Settings → Secrets**, add:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

---

## 📁 File Structure

```
recallbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .env                # Local secrets (never commit!)
├── recallbot.db        # SQLite database (auto-created)
└── chroma_store/       # ChromaDB vector store (auto-created)
```

---

## 🔐 Demo Account

A demo account is automatically created on first run:

| Field | Value |
|-------|-------|
| Username | `demo` |
| Password | `demo123` |

---

## 📖 How to Export Chats

### WhatsApp
1. Open a chat → tap ⋮ (Android) or name (iOS)
2. **More → Export Chat → Without Media**
3. Save the `.txt` file and upload to RecallBot

### Slack
1. Go to **Workspace Settings → Import/Export**
2. Click **Export** → **All time** → **Start Export**
3. Download the `.zip` file and upload to RecallBot

---

## 🏗️ Tech Stack

- **Frontend**: Streamlit with custom CSS (glassmorphism, dark/light mode)
- **AI**: OpenAI GPT-4o-mini (streaming responses)
- **Vector DB**: ChromaDB (persistent embeddings)
- **Embeddings**: `all-MiniLM-L6-v2` via SentenceTransformers
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
- **Storage**: SQLite (users, messages, saved queries)
- **Analytics**: Plotly (bar charts, line charts, heatmaps)
- **Auth**: SHA-256 password hashing

---

## ⚙️ Configuration

All config constants are at the top of `app.py`:

```python
EMBED_MODEL   = "all-MiniLM-L6-v2"   # Embedding model
GPT_MODEL     = "gpt-4o-mini"         # OpenAI model
CHUNK_SIZE    = 400                    # Token chunk size
CHUNK_OVERLAP = 60                     # Overlap between chunks
MAX_RESULTS   = 6                      # Semantic search results
```

---

## 📄 License

MIT © 2024 RecallBot. Built with ❤️ using Streamlit and OpenAI.
