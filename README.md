# 🐦 Twitter Sentiment Analyzer
### Full-Stack ML Deployment — Internship Capstone Project

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)](https://render.com)

---

## 🔗 Live URLs

| Service | URL |
|---------|-----|
| **Frontend UI** | https://sentiment-frontend-no7g.onrender.com |
| **Backend API** | https://twitter-sentiment-xegm.onrender.com |
| **API Docs (Swagger)** | https://twitter-sentiment-xegm.onrender.com/docs |

---

## 📌 Project Overview

**Twitter Sentiment Analyzer** is a production-grade full-stack ML application that classifies tweets and short-form text into **Positive**, **Negative**, or **Neutral** sentiments in real time.

Built as an internship capstone project to demonstrate the complete journey from a trained ML model in a Jupyter notebook to a live, publicly accessible web application.

### What makes this different from a basic project

- **Three models evaluated** — TF-IDF + Logistic Regression, DistilBERT, and BERTweet. Deployed TF-IDF for production due to memory constraints on free cloud infrastructure
- **Overfitting detected and fixed** — training/test gap reduced from 7.9% to 2.74%
- **Enhanced NLP pipeline** — emoji conversion, slang dictionary, negation handling, lemmatization, language detection
- **Live news analysis** — fetches real headlines via NewsAPI and analyzes sentiment in real time
- **Reliability scoring** — every prediction comes with a reliability label (High/Medium/Low) based on confidence and sarcasm detection
- **Batch prediction endpoint** — process up to 50 texts in a single API call
- **UptimeRobot monitoring** — backend kept alive 24/7 on free tier

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Scikit-learn — TF-IDF Vectorizer + Logistic Regression |
| **NLP Pipeline** | NLTK — tokenization, lemmatization, POS tagging, stopwords |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend UI** | Streamlit + Plotly |
| **News Integration** | NewsAPI via requests |
| **Emoji Handling** | emoji library (demojize) |
| **Language Detection** | langdetect |
| **Serialization** | joblib (.pkl files) |
| **Containerization** | Docker |
| **Cloud Hosting** | Render (free tier) |
| **Uptime Monitoring** | UptimeRobot |
| **Version Control** | Git + GitHub |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF (75K features, bigrams, sublinear_tf) |
| Training samples | 200,000 tweets |
| Test accuracy | **77.00%** |
| Train accuracy | 80.68% |
| Overfitting gap | **2.74%** (healthy) |
| Baseline (random) | 50% |

### Models evaluated

| Model | Accuracy | Size | Deployed |
|-------|----------|------|----------|
| TF-IDF + Logistic Regression | 77.00% | 21 MB | ✅ Yes |
| DistilBERT | 81.30% | 268 MB | ❌ Memory constraints |
| BERTweet | ~85-88% | 500+ MB | ❌ Memory constraints |

**Deployment decision:** TF-IDF chosen for production — instant inference, 21MB model size fits free cloud tier, no GPU required.

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server liveness check |
| GET | `/stats` | Usage statistics dashboard |
| POST | `/predict` | Single text sentiment analysis |
| POST | `/batch_predict` | Batch analysis (up to 50 texts) |
| POST | `/topic_sentiment` | Live news sentiment by topic |

### Example — POST /predict

**Request:**
```json
{
  "text": "I absolutely love this new phone, it's incredible"
}
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "input_text": "I absolutely love this new phone, it's incredible",
  "char_count": 49,
  "language": "English",
  "overall_sentiment": "positive",
  "overall_confidence": 0.9433,
  "sentence_count": 1,
  "sentences": [
    {
      "sentence": "I absolutely love this new phone, it's incredible",
      "sentiment": "positive",
      "confidence": 0.9433
    }
  ],
  "model_version": "2.0.0",
  "processing_time_ms": 18.5,
  "reliability_score": "High (Good confidence)",
  "warning": null
}
```

---

## 🗂 Project Structure

```
twitter-sentiment/
├── app/
│   ├── main.py              # FastAPI application — 5 endpoints
│   └── utils.py             # Text cleaning utilities
├── frontend/
│   └── app.py               # Streamlit UI — 3 pages
├── saved_model/
│   ├── sentiment_model.pkl  # Trained Logistic Regression
│   ├── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
│   └── utils.py             # Shared NLP pipeline
├── model/
│   └── sentiment_training.ipynb  # Full training notebook
├── Dockerfile               # Container build instructions
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes .env, __pycache__
└── README.md
```

---

## ⚙️ NLP Pipeline

Every tweet goes through this pipeline before prediction:

```
Raw text
  → Emoji conversion (😭 → "loudly_crying_face")
  → Slang replacement ("bussin" → "amazing")
  → Apostrophe normalization ("don't" → "dont")
  → Negation handling ("dont love" → "not_love")
  → Lowercase
  → Number removal
  → Punctuation removal
  → Stopword removal (198 words)
  → Short word filter (< 3 chars)
  → Lemmatization with POS tagging
  → TF-IDF vectorization (75K features)
  → Logistic Regression prediction
  → Confidence threshold (< 65% → Neutral)
  → Reliability scoring
```

---

## 🎯 Reliability Score Guide

| Score | Meaning |
|-------|---------|
| `High (Good confidence)` | Confidence ≥ 85% — strong prediction |
| `Medium (Moderate confidence)` | Confidence 75–84% |
| `Medium (Uncertain confidence)` | Confidence 65–74% |
| `Low (Low confidence)` | Confidence < 65% — model is unsure |
| `Low (Short text)` | Fewer than 4 words |
| `Low (Sarcasm detected)` | Sarcasm signals found — prediction may be wrong |

---

## ⚠️ Known Limitations

| Limitation | Reason | Fix |
|-----------|--------|-----|
| Sarcasm detection unreliable | TF-IDF has no contextual understanding | Use BERTweet |
| Double negation fails | "not bad" → model sees "bad" | Transformer architecture |
| English only | Trained on English tweets | Multilingual model |
| Emoji vocabulary limited | Rare in training data | Larger training sample |
| Cold start delay | Free tier spin-down | Mitigated by UptimeRobot |

---

## 🏃 Run Locally

### Prerequisites
- Python 3.12+
- NewsAPI key (free at newsapi.org)

### 1 — Clone
```bash
git clone https://github.com/Naina2006-Vismi/twitter-sentiment.git
cd twitter-sentiment
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Configure environment
Create `.env` file:
```
NEWS_API_KEY=your_key_here
```

### 4 — Start backend
```bash
uvicorn app.main:app --reload --port 8000
```

### 5 — Start frontend
```bash
streamlit run frontend/app.py
```

### 6 — Run with Docker
```bash
docker build -t twitter-sentiment .
docker run -p 8000:8000 --env-file .env twitter-sentiment
```

---

## 📈 Engineering Decisions

### Why TF-IDF over BERT?

| Factor | TF-IDF + LR | BERT/DistilBERT |
|--------|-------------|-----------------|
| Model size | 21 MB | 268–500 MB |
| Inference speed | < 50ms | 200–500ms |
| Free tier RAM (512MB) | ✅ Fits | ❌ Crashes |
| Accuracy | 77% | 81–88% |
| Explainability | High | Low |

TF-IDF was chosen because it fits within free cloud infrastructure constraints while delivering acceptable accuracy. DistilBERT was evaluated (81.3%) and documented but not deployed.

### Why FastAPI over Flask?

- Auto-generated Swagger UI at `/docs`
- Built-in Pydantic validation
- Better performance with async support
- Modern Python type hints

### Why Render over AWS?

- Zero configuration for Docker deployment
- Free tier sufficient for demonstration
- Auto-deploy on GitHub push (CI/CD built-in)
- No credit card required

---

## 👩‍💻 Developed By

**Nainavismi**
Internship Capstone Project — 2026

---

*Built with FastAPI + Streamlit + Docker + Render*
