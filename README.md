# 🎓 AI Video Learning Assistant

A full-stack AI-powered web application that transforms YouTube videos into structured learning content — **summaries, key points, detailed notes, and interactive quizzes**.

Built with **React + Tailwind CSS** (frontend) and **Flask + Hugging Face Transformers** (backend).

---

## ✨ Features

- 🔗 **YouTube URL Input** — Paste any YouTube link
- 📝 **Auto Transcript** — Fetches captions (with Whisper fallback)
- 🤖 **AI Summary** — Concise summary powered by BART
- 💡 **Key Points** — Bullet-point insights via Flan-T5
- 📚 **Study Notes** — Detailed, organized notes
- 🧠 **Interactive Quiz** — MCQ quiz with instant feedback & scoring
- 📋 **Copy to Clipboard** — One-click copy for any section
- 📄 **Download PDF** — Export notes as PDF
- 🕐 **Search History** — LocalStorage-based video history
- 🌙 **Dark/Light Mode** — Toggle between themes
- ⚡ **Premium UI** — Glassmorphism, animations, responsive design

---

## 📁 Project Structure

```
AI Tools Project/
├── backend/
│   ├── app.py                     # Flask server entry point
│   ├── requirements.txt           # Python dependencies
│   ├── utils/
│   │   ├── __init__.py
│   │   └── video_utils.py         # YouTube ID extraction & metadata
│   └── services/
│       ├── __init__.py
│       ├── transcript_service.py  # Transcript fetching (captions + Whisper)
│       ├── whisper_service.py     # Whisper speech-to-text fallback
│       └── llm_service.py         # AI pipeline (BART + Flan-T5)
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css              # Design system (dark/light mode)
│       ├── services/
│       │   └── api.js             # Axios API service
│       └── components/
│           ├── Navbar.jsx
│           ├── InputForm.jsx
│           ├── LoadingState.jsx
│           ├── ResultsDashboard.jsx
│           ├── SummaryCard.jsx
│           ├── KeyPointsCard.jsx
│           ├── NotesCard.jsx
│           ├── QuizSection.jsx
│           └── HistorySidebar.jsx
└── README.md
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **pip** (Python package manager)

### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

The backend will start on **http://localhost:5000**.

> ⚠️ **First request will be slow** — the AI models (BART + Flan-T5) are downloaded on first use (~1-2 GB). Subsequent requests are fast.

### 2. Frontend Setup

```bash
# Navigate to frontend (in a new terminal)
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start on **http://localhost:3000**.

### 3. Usage

1. Open **http://localhost:3000** in your browser
2. Paste a YouTube video URL
3. Click **"Analyze Video"**
4. Wait for AI processing (1-3 minutes on first run)
5. Explore your **Summary**, **Key Points**, **Notes**, and take the **Quiz**!

---

## 🛠 Tech Stack

| Layer      | Technology                              |
| ---------- | --------------------------------------- |
| Frontend   | React, Vite, Tailwind CSS, Framer Motion |
| Backend    | Python, Flask, Flask-CORS               |
| AI Models  | BART (summarization), Flan-T5 (generation) |
| Transcript | youtube-transcript-api, OpenAI Whisper  |
| Video API  | YouTube Data API v3                     |

---

## 📡 API Endpoints

| Method | Endpoint              | Description                      |
| ------ | --------------------- | -------------------------------- |
| GET    | `/api/health`         | Health check                     |
| POST   | `/api/summarize-video`| Full AI analysis of a video      |
| POST   | `/api/transcript`     | Fetch transcript only            |

### Example Request

```bash
curl -X POST http://localhost:5000/api/summarize-video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

---

## 📄 License

MIT License — Free for educational and personal use.
