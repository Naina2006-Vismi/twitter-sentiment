import streamlit as st
import requests
import plotly.express as px
import pandas as pd


# ─── Auto-wake backend on startup ──────────────────────────────
def wake_backend():
    try:
        response = requests.get(f"{API_URL}/health", timeout=90)
        if response.status_code == 200:
            return True
    except:
        return False

with st.spinner("🔄 Connecting to backend... (may take 30-60 seconds on first load)"):
    backend_alive = wake_backend()

if not backend_alive:
    st.warning("⚠️ Backend is still warming up. Please refresh the page in 30 seconds.")
    st.info("💡 Free tier servers sleep after inactivity. First load takes 60-90 seconds.")
    st.stop()

# ─── Config ────────────────────────────────────────────────────
API_URL = "https://twitter-sentiment-xegm.onrender.com"

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    layout="wide"
)

# ─── Sidebar ───────────────────────────────────────────────────
st.sidebar.title(" Twitter Sentiment Analyzer")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Tweet Sentiment", "Live News Analyzer", "API Stats"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Nainavismi**")
st.sidebar.markdown("Powered by Streamlit")
st.sidebar.markdown("**Version:** 2.0.0")

# ─── Helpers ───────────────────────────────────────────────────
def sentiment_color(sentiment):
    icons = {
        "positive": "🟢",
        "negative": "🔴",
        "neutral" : "🟡"
    }
    return icons.get(sentiment.lower(), "⚪")

def call_api(endpoint: str, payload: dict):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        return response.json(), None
    except requests.ConnectionError:
        return None, "Cannot connect to API. Make sure the backend is running on port 8000."
    except requests.Timeout:
        return None, "Request timed out. Please try again."
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get('detail', str(e))
        except Exception:
            detail = str(e)
        return None, f"API error: {detail}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Tweet Sentiment
# ══════════════════════════════════════════════════════════════
if page == "Tweet Sentiment":
    st.title("Tweet Sentiment Analysis")
    st.markdown("Enter a tweet or any short text to analyze its sentiment.")
    st.markdown("---")

    tweet_input = st.text_area(
        "Enter your tweet:",
        placeholder="Type something like: I love this new phone!",
        height=120,
        max_chars=2800
    )
    char_count = len(tweet_input)
    st.caption(f"Character count: {char_count}/2800")

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button(
            "Analyze Sentiment ",
            type="primary",
            use_container_width=True
        )
    with col2:
        clear_button = st.button("Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    if analyze_button:
        if not tweet_input.strip():
            st.warning("Please enter a tweet to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                data, error = call_api("predict", {"text": tweet_input})

            if error:
                st.error(error)
            else:
                st.markdown("---")

                sentiment  = data['overall_sentiment']
                confidence = data['overall_confidence']
                icon       = sentiment_color(sentiment)

                # Main metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Overall Sentiment",
                        value=f"{icon} {sentiment.capitalize()}"
                    )
                with col2:
                    st.metric(
                        label="Confidence",
                        value=f"{confidence*100:.1f}%"
                    )
                with col3:
                    st.metric(
                        label="Sentences Analyzed",
                        value=data['sentence_count']
                    )

                # Reliability score
                st.info(f"Reliability: {data['reliability_score']}")

                # Warning
                if data.get('warning'):
                    st.warning(f" {data['warning']}")

                # Sentence breakdown
                if data['sentence_count'] > 1:
                    st.markdown("#### Sentence-level Breakdown")
                    for sentence in data['sentences']:
                        icon_s = sentiment_color(sentence['sentiment'])
                        with st.expander(
                            f"{icon_s} {sentence['sentence'][:80]}"
                        ):
                            st.write(f"**Sentiment:** {sentence['sentiment'].capitalize()}")
                            st.write(f"**Confidence:** {sentence['confidence']*100:.1f}%")
                            st.progress(sentence['confidence'])

                # Metadata
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Request ID: `{data['request_id']}`")
                with col2:
                    st.caption(f"Processing Time: {data['processing_time_ms']}ms")
                with col3:
                    st.caption(f"Characters: {data['char_count']}")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Live News Analyzer
# ══════════════════════════════════════════════════════════════
elif page == "Live News Analyzer":
    st.title("Live News Sentiment Analyzer")
    st.markdown("Analyze the sentiment of the latest news on any topic.")
    st.markdown("---")

    topic_input = st.text_input(
        "Enter a news topic:",
        placeholder="e.g. Tesla, OpenAI, climate change, iPhone"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        news_button = st.button(
            "Analyze News",
            type="primary",
            use_container_width=True
        )

    if news_button:
        if not topic_input.strip():
            st.warning("Please enter a topic to analyze.")
        else:
            with st.spinner(f"Fetching latest news about '{topic_input}'..."):
                data, error = call_api(
                    "topic_sentiment",
                    {"topic": topic_input, "count": 5}
                )

            if error:
                st.error(error)
            else:
                st.markdown("---")

                sentiment = data['overall_sentiment']
                icon      = sentiment_color(sentiment)

                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Overall Sentiment",
                        f"{icon} {sentiment.capitalize()}"
                    )
                with col2:
                    st.metric("🟢 Positive", data['positive_count'])
                with col3:
                    st.metric("🔴 Negative", data['negative_count'])
                with col4:
                    st.metric("🟡 Neutral",  data['neutral_count'])

                # Pie chart
                chart_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count"    : [
                        data['positive_count'],
                        data['negative_count'],
                        data['neutral_count']
                    ]
                })
                fig = px.pie(
                    chart_data,
                    values="Count",
                    names="Sentiment",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#2ecc71",
                        "Negative": "#e74c3c",
                        "Neutral" : "#f39c12"
                    },
                    title=f"Sentiment Distribution for '{topic_input}'"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Article cards
                st.markdown("### Latest Headlines")
                for article in data['articles']:
                    icon_a = sentiment_color(article['sentiment'])
                    with st.expander(
                        f"{icon_a} {article['sentiment'].capitalize()} "
                        f"({article['confidence']*100:.0f}%) — "
                        f"{article['title'][:70]}"
                    ):
                        st.markdown(f"**Headline:** {article['title']}")
                        if article.get('description'):
                            st.markdown(f"**Summary:** {article['description']}")
                        st.markdown(f"**Source:** {article['source']}")
                        st.markdown(f"**Published:** {article['publishedAt'][:10]}")
                        st.markdown(f"**Confidence:** {article['confidence']*100:.1f}%")
                        st.progress(article['confidence'])
                        st.markdown(f"[Read full article →]({article['url']})")

                # Metadata
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Request ID: `{data['request_id']}`")
                with col2:
                    st.caption(f"Processing Time: {data['processing_time_ms']}ms")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — API Stats
# ══════════════════════════════════════════════════════════════
elif page == "API Stats":
    st.title(" API Usage Statistics")
    st.markdown("Live usage statistics from the sentiment analysis API.")
    st.markdown("---")

    st.button("Refresh Stats", type="primary")

    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        stats    = response.json()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Requests", stats['total_requests'])
        with col2:
            st.metric(
                "Avg Confidence",
                f"{stats['average_confidence']*100:.1f}%"
            )
        with col3:
            st.metric("Uptime Since", stats['uptime'][:10])

        if stats['sentiment_counts']:
            st.markdown("### Sentiment Breakdown")
            df = pd.DataFrame({
                "Sentiment": list(stats['sentiment_counts'].keys()),
                "Count"    : list(stats['sentiment_counts'].values())
            })
            fig = px.bar(
                df,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map={
                    "positive": "#2ecc71",
                    "negative": "#e74c3c",
                    "neutral" : "#f39c12"
                },
                title="Total Predictions by Sentiment"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No requests yet. Use the Tweet Analyzer or News Analyzer first.")

    except requests.ConnectionError:
        st.error("Cannot connect to API. Make sure backend is running on port 8000.")
    except Exception as e:
        st.error(f"Error fetching stats: {str(e)}")