from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import sys
import os
import time
import uuid
import requests
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
saved_model_path = os.path.join(base_dir, '..','saved_model')
sys.path.insert(0, saved_model_path)

from utils import (
    clean_text_v2,
    detect_language,
    analyze_sentences,
    is_likely_neutral,
    get_reliability_score
)

load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing the sentiment of tweets and short text.",
    version="2.0.0"
)

print("Loading model and vectorizer...")
model = joblib.load(os.path.join(saved_model_path, 'sentiment_model.pkl'))
vectorizer = joblib.load(os.path.join(saved_model_path, 'tfidf_vectorizer.pkl'))
print("Model and vectorizer loaded successfully.")

stats = {
    'total_requests': 0,
    'sentiment_counts': defaultdict(int),
    'total_confidence': 0.0,
    'start_time': datetime.now().isoformat()
}

from pydantic import BaseModel, field_validator

class TweetRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError('Text cannot be empty')
        if len(value) > 2800:
            raise ValueError('Text too long. Maximum 2800 characters.')
        return value

class BatchRequest(BaseModel):
    texts: list[str]

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, value):
        if not value:
            raise ValueError('texts list cannot be empty')
        if len(value) > 50:
            raise ValueError('Maximum 50 texts per batch request')
        return value

class TopicRequest(BaseModel):
    topic: str
    count: int = 5

    @field_validator('topic')
    @classmethod
    def validate_topic(cls, value):
        if not value.strip():
            raise ValueError('Topic cannot be empty')
        if len(value) > 100:
            raise ValueError('Topic too long')
        return value.strip()

    @field_validator('count')
    @classmethod
    def validate_count(cls, value):
        if value < 1:
            raise ValueError('Count must be at least 1')
        if value > 10:
            raise ValueError('Maximum 10 articles per request')
        return value

class SentenceResult(BaseModel):
    sentence: str
    sentiment: str
    confidence: float

class ArticleResult(BaseModel):
    title: str
    description: str
    source: str
    url: str
    publishedAt: str
    sentiment: str
    confidence: float

class SentimentResponse(BaseModel):
    model_config = {'protected_namespaces': ()} 
    request_id: str
    input_text: str
    char_count: int
    language: str
    overall_sentiment  : str
    overall_confidence : float
    sentence_count     : int
    sentences          : list[SentenceResult]
    model_version      : str
    processing_time_ms : float
    reliability_score   : str 
    warning            : str | None = None
    
class BatchSentimentResponse(BaseModel):
    request_id: str
    total_texts: int
    results: list[SentimentResponse]
    processing_time_ms: float
    
class TopicsSentimentResponse(BaseModel):
    request_id: str
    topic: str
    total_articles: int
    overall_sentiment: str
    positive_count: int
    negative_count: int
    neutral_count: int
    articles: list[ArticleResult]
    processing_time_ms: float

def single_predict(text: str):
    if is_likely_neutral(text):
        return 'neutral', 1.0
    
    cleaned_text = clean_text_v2(text)

    if not cleaned_text.strip():
        return 'neutral', 0.0
    
    vectorized_text = vectorizer.transform([cleaned_text])
    sentiment_prob = model.predict_proba(vectorized_text)[0]
    predicted_label = model.predict(vectorized_text)[0]
    confidence = float(sentiment_prob[predicted_label])

    if confidence < 0.65:
        sentiment = 'neutral'
    elif predicted_label == 1:
        sentiment = 'positive'
    else:
        sentiment = 'negative'

    return sentiment, round(confidence, 4)

def build_response(text: str, request_id: str, start: float):
    is_english = detect_language(text)
    if not is_english:
        raise HTTPException(status_code=400, detail="Input text must be in English.")
    
    sentences = analyze_sentences(text)
    if not sentences:
        sentences = [text]

    sentence_results = []
    for sentence in sentences:
        sentiment, confidence = single_predict(sentence)
        sentence_results.append(SentenceResult(sentence=sentence, sentiment=sentiment, confidence=confidence))

    sentiments = [res.sentiment for res in sentence_results]
    postive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')

    if postive_count > negative_count:
        overall_sentiment = 'positive'
    elif negative_count > postive_count:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'
    overall_confidence = round(sum(res.confidence for res in sentence_results) / len(sentence_results), 4)

    stats['total_requests'] += 1
    stats['sentiment_counts'][overall_sentiment] += 1
    stats['total_confidence'] += overall_confidence

    processing_time_ms = round((time.time() - start) * 1000, 2)

    warning = None
    if len(sentence_results) > 1:
        warning = "Multiple sentences detected. Sentence-level breakdown provided."  

    reliability_score = get_reliability_score(text, overall_confidence)
    return SentimentResponse(
        request_id=request_id,
        input_text=text,
        char_count=len(text),
        language='English',
        overall_sentiment=overall_sentiment,
        overall_confidence=overall_confidence,
        sentence_count=len(sentence_results),
        sentences=sentence_results,
        model_version='2.0.0',
        processing_time_ms=processing_time_ms,
        reliability_score = reliability_score,
        warning=warning
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "model" : "TF-IDF + Logistic Regression", "model_version": "2.0.0", "uptime": stats['start_time']}

@app.get("/stats")
def get_stats():
    total_requests = stats['total_requests']
    average_confidence = round(stats['total_confidence'] / total_requests, 4) if total_requests > 0 else 0.0
    return {
        "total_requests": stats['total_requests'],
        "sentiment_counts": dict(stats['sentiment_counts']),
        "average_confidence": average_confidence,
        "uptime": stats['start_time']
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: TweetRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    return build_response(request.text, request_id, start_time)

@app.post("/batch_predict", response_model=BatchSentimentResponse)
def batch_predict(request: BatchRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    results = []
    for text in request.texts:
        try:
            inner_start = time.time()
            result = build_response(text, request_id, inner_start)
            results.append(result)
        except HTTPException as e:
            results.append(SentimentResponse(
                request_id=request_id,
                input_text=text,
                char_count=len(text),
                language='Non-English or Invalid',
                overall_sentiment='error',
                overall_confidence=0.0,
                sentence_count=0,
                sentences=[],
                model_version='2.0.0',
                processing_time_ms=round((time.time() - inner_start) * 1000, 2),
                warning= 'Non-English text or other validation error: ' + str(e.detail)
            ))
    processing_time_ms = round((time.time() - start_time) * 1000, 2)

    return BatchSentimentResponse(
        request_id=request_id,
        total_texts=len(request.texts),
        results=results,
        processing_time_ms=processing_time_ms
    )

@app.post("/topic_sentiment", response_model=TopicsSentimentResponse)
def topic_sentiment(request: TopicRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    if not NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="News API key not configured.")

    try:
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': request.topic,
                'pageSize': request.count,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': NEWS_API_KEY
            },
            timeout=10
        )
        response.raise_for_status()
        news_data = response.json()
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="News API request timed out.")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error fetching news articles: {str(e)}")

    articles = news_data.get('articles', [])
    if not articles:
        raise HTTPException(status_code=404, detail="No news articles found for the given topic.")

    results = []

    for article in articles:
        heading = article.get('title', '')
        source = article.get('source', {}).get('name', 'Unknown Source')
        url = article.get('url', '')
        published_at = article.get('publishedAt', '')
        description = article.get('description', '')
    
        if not heading or heading == 'null':
            continue
        sentiment, confidence = single_predict(heading)

        results.append(ArticleResult(
            title =heading,
            description=description,
            source=source,
            url=url,
            publishedAt=published_at,
            sentiment=sentiment,
            confidence=confidence
        ))

    if not results:
        raise HTTPException(status_code=404, detail="No valid news articles found for sentiment analysis.")

    sentiments = [res.sentiment for res in results]
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    if positive_count > negative_count:
        overall_sentiment = 'positive'
    elif negative_count > positive_count:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'

    stats['total_requests'] += 1
    stats['sentiment_counts'][overall_sentiment] += 1
    processing_time_ms = round((time.time() - start_time) * 1000, 2)

    return TopicsSentimentResponse(
        request_id = request_id,
        topic = request.topic,
        total_articles = len(results),
        overall_sentiment = overall_sentiment,
        positive_count = positive_count,
        negative_count = negative_count,
        neutral_count = neutral_count,
        articles = results,
        processing_time_ms = processing_time_ms
    )
