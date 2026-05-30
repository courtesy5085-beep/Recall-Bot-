# 🧠 RecallBot — AI-Powered Memory Assistant for Team Chats

> *Your conversations, remembered forever. Powered by GPT-4o-mini + ChromaDB.*

---

## ✨ Features

- 📁 **Import** WhatsApp `.txt` & Slack `.json`/`.zip` exports
- 🧠 **AI Memory Engine** — ChromaDB + SentenceTransformers + BM25 hybrid search
- 💬 **Ask anything** — streaming GPT-4o-mini answers with cited sources
- 📊 **Analytics Dashboard** — Plotly charts, heatmaps, activity trends
- 🔖 **Bookmarks & Saved Answers** — pin important memories
- 🔐 **Secure Auth** — SQLite + PBKDF2-SHA256 password hashing
- 🎨 **Premium Futuristic UI** — glassmorphism, neon gradients, animations

---

## 🚀 Quick Start

### Local Setup

```bash
git clone https://github.com/yourname/recallbot.git
cd recallbot
pip install -r requirements.txt
streamlit run app.py
```

### Add your OpenAI key

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

Or in Streamlit Cloud: go to **App Settings → Secrets** and add:
```toml
OPENAI_API_KEY = "sk-..."
```

---

## ☁️ Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set `app.py` as main file
5. Add `OPENAI_API_KEY` in Secrets
6. Deploy!

---

## 🗂️ File Structure

```
recallbot/
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml        # Theme config
├── .gitignore
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS |
| AI | GPT-4o-mini (OpenAI) |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers |
| Search | BM25 + Semantic Hybrid |
| Auth | SQLite + bcrypt |
| Charts | Plotly |

---

## 📄 License

MIT License — use freely, attribution appreciated.
