# 🐦 Twitter Persevex — Tweet Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 About the Project

**Twitter Persevex** is a full-stack sentiment analysis application that classifies tweets and short-form text into **Positive**, **Negative**, or **Neutral** sentiments in real time.

### Why does this exist?

Understanding public sentiment around topics — whether a brand, a news story, or a social issue — has real value. Existing tools are often heavyweight, API-rate-limited, or black-box. Persevex is lightweight, self-hosted, and fully transparent: a trained TF-IDF + Logistic Regression model serves predictions through a FastAPI backend, with a polished Streamlit frontend that requires zero frontend experience to run or extend.

### Key Highlights

- **Tweet Sentiment Analyzer** — Classify any English tweet or short text with sentence-level breakdown and a reliability score
- **Live News Analyzer** — Fetch the latest news articles by topic (via NewsAPI) and visualize their aggregate sentiment with an interactive pie chart
- **API Stats Dashboard** — Live bar charts and metrics of all predictions made since the server started
- **Slang & Emoji Awareness** — A curated slang dictionary (e.g. `"bussin"` → `"amazing"`) and emoji-to-text conversion keep Gen-Z tweets correctly classified
- **Negation Handling** — Tokens like `"not_good"` are preserved through the pipeline so negations don't silently flip sentiment
- **Batch Endpoint** — Submit up to 50 texts in a single API call

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **Frontend UI** | [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/python/) |
| **ML Model** | Scikit-learn — TF-IDF Vectorizer + Logistic Regression |
| **NLP Pipeline** | NLTK (tokenization, lemmatization, POS tagging, stopwords) |
| **Emoji Handling** | `emoji` library (demojize) |
| **Language Detection** | `langdetect` |
| **News Integration** | [NewsAPI](https://newsapi.org/) via `requests` |
| **Config Management** | `python-dotenv` |
| **Model Serialization** | `joblib` (`.pkl` files) |
| **Language** | Python 3.11+ |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11** or higher
- A free [NewsAPI](https://newsapi.org/register) API key (required for the Live News Analyzer feature)
- `pip` (comes bundled with Python)

### 1 — Clone the Repository

```bash
git clone https://github.com/your-username/twitter-persevex.git
cd twitter-persevex
```

### 2 — Create and Activate a Virtual Environment

```bash
# Create the environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** NLTK corpora (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`) are downloaded automatically on the first run.

### 4 — Configure Environment Variables

Create a `.env` file in the project root (or rename the existing one):

```bash
cp .env.example .env   # if an example file exists, otherwise create manually
```

Then open `.env` and add your NewsAPI key:

```ini
NEWS_API_KEY=your_newsapi_key_here
```

> The `.env` file is already listed in `.gitignore` — your key will **not** be committed.

### 5 — Start the Backend API

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive Swagger docs are auto-generated at `http://localhost:8000/docs`.

### 6 — Start the Frontend

Open a **second terminal** (with the virtual environment activated) and run:

```bash
streamlit run frontend/app.py
```

The Streamlit UI will open automatically in your browser at `http://localhost:8501`.

---

## 📖 Usage

### Via the Streamlit UI

Navigate between three pages using the sidebar:

- **Tweet Sentiment** — Paste any tweet (up to 2800 characters) and click *Analyze Sentiment*
- **Live News Analyzer** — Enter a topic keyword (e.g. `Tesla`, `climate change`) and click *Analyze News*
- **API Stats** — View a live dashboard of total requests, average confidence, and sentiment distribution

---

### Via the REST API

#### `POST /predict` — Single Text Analysis

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this new update, it slaps!"}'
```

**Example Response:**

```json
{
  "request_id": "a1b2c3d4",
  "input_text": "I absolutely love this new update, it slaps!",
  "char_count": 44,
  "language": "English",
  "overall_sentiment": "positive",
  "overall_confidence": 0.9312,
  "sentence_count": 1,
  "sentences": [
    {
      "sentence": "I absolutely love this new update, it slaps!",
      "sentiment": "positive",
      "confidence": 0.9312
    }
  ],
  "model_version": "2.0.0",
  "processing_time_ms": 14.5,
  "reliability_score": "High (Good confidence)",
  "warning": null
}
```

---

#### `POST /batch_predict` — Batch Analysis (up to 50 texts)

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This product is amazing!",
      "Worst experience I have ever had.",
      "The package arrived on Tuesday."
    ]
  }'
```

---

#### `POST /topic_sentiment` — Live News Sentiment by Topic

```bash
curl -X POST "http://localhost:8000/topic_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"topic": "OpenAI", "count": 5}'
```

**Example Response (abbreviated):**

```json
{
  "request_id": "f9e8d7c6",
  "topic": "OpenAI",
  "total_articles": 5,
  "overall_sentiment": "positive",
  "positive_count": 3,
  "negative_count": 1,
  "neutral_count": 1,
  "articles": [ ... ],
  "processing_time_ms": 423.7
}
```

---

#### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "TF-IDF + Logistic Regression",
  "model_version": "2.0.0",
  "uptime": "2025-05-28T19:44:00.000000"
}
```

---

#### `GET /stats` — Usage Statistics

```bash
curl http://localhost:8000/stats
```

```json
{
  "total_requests": 42,
  "sentiment_counts": { "positive": 22, "negative": 13, "neutral": 7 },
  "average_confidence": 0.8471,
  "uptime": "2025-05-28T19:44:00.000000"
}
```

---

### Reliability Score Guide

| Score | Meaning |
|---|---|
| `High (Good confidence)` | Confidence ≥ 85% — strong prediction |
| `Medium (Moderate confidence)` | Confidence 75–84% — reasonable prediction |
| `Medium (Uncertain confidence)` | Confidence 65–74% — treat with care |
| `Low (Low confidence)` | Confidence < 65% — model is unsure |
| `Low (Short text)` | Fewer than 4 words — insufficient signal |
| `Low (Sarcasm detected)` | Sarcasm signals found — prediction may be inverted |

---

## 🗂 Project Structure

```
twitter-persevex/
├── app/
│   ├── main.py              # FastAPI application, routes, and response models
│   └── utils.py             # Text cleaning, negation handling, slang replacement
├── frontend/
│   └── app.py               # Streamlit UI (3 pages)
├── saved_model/
│   ├── sentiment_model.pkl  # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
│   └── utils.py             # Shared NLP utilities + reliability scoring
├── model/
│   └── sentiment_training.ipynb  # Model training notebook
├── requirements.txt
├── .env                     # API keys (not committed)
└── .gitignore
```

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Here's how to get involved:

1. **Fork** the repository
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your changes** with a clear message
   ```bash
   git commit -m "feat: add sarcasm detection layer"
   ```
4. **Push** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** against the `main` branch

### Guidelines

- Keep PRs focused — one feature or fix per PR
- Add or update docstrings for any new functions
- If you change the NLP pipeline, run the training notebook and attach updated model metrics
- Non-English support is a known limitation and a welcome area for contribution

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Nainavismi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">Built with ❤️ by <strong>Nainavismi</strong> · Powered by FastAPI & Streamlit</p>
