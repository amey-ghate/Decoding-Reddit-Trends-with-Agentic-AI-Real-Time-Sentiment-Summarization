import os
import json
from typing import List, TypedDict

import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

# -----------------------------------------------------------------------------
# Streamlit & Matplotlib setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Reddit Sentiment Agent", layout="wide")
plt.style.use("ggplot")

# -----------------------------------------------------------------------------
# Load VADER / sentiment model from pickle
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    with open("model_pickle", "rb") as f:
        model = pickle.load(f)
    return model

model = load_sentiment_model()

# -----------------------------------------------------------------------------
# Session state init (store last analysis so selectbox doesn’t lose it)
# -----------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state["analysis"] = None

# -----------------------------------------------------------------------------
# OpenAI client setup (.env)
# -----------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your .env file (OPENAI_API_KEY=sk-...)")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# LangGraph state and nodes
# -----------------------------------------------------------------------------
class CommentPipelineState(TypedDict):
    comments: List[str]
    cleaned_comments: List[str]
    summary: str


CLEANING_INSTRUCTIONS = """
You are a helpful assistant that cleans Reddit comments for sentiment analysis.

Given an ARRAY of Reddit comments:
- Remove URLs, markdown artifacts, user mentions, and obvious spam fragments.
- Remove emojis and non-text noise.
- Expand common contractions (don't -> do not, can't -> cannot) where reasonable.
- Preserve the original meaning and sentiment as much as possible.
- Keep profanity if it is important for sentiment.
- DO NOT translate to another language.
- IMPORTANT: Return ONLY a valid JSON array of cleaned comment strings, in the SAME ORDER as input.
  No extra text, no explanations, no keys, just the JSON array.
"""

SUMMARY_INSTRUCTIONS = """
You are a data analyst summarizing online discussions.

Given a sample of cleaned Reddit comments about a set of posts, write a concise
3–6 sentence summary that includes:

- The main topics people are discussing overall.
- The overall sentiment (positive, negative, or mixed).
- Any noticeable patterns, disagreements, or strong emotions.

Write in neutral, clear English. Do not list every single comment; instead, describe patterns.
"""


def clean_comments_with_llm(comments: List[str], batch_size: int = 20) -> List[str]:
    """
    Use OpenAI chat completions to clean comments in batches.
    Returns cleaned comments aligned with the input order.
    """
    if not comments:
        return []

    cleaned_all: List[str] = []

    for start in range(0, len(comments), batch_size):
        batch = comments[start:start + batch_size]

        prompt = f"""
You will receive a JSON array of Reddit comments under the key "comments".
Clean them according to the earlier instructions and return ONLY a JSON array of cleaned comments.

Input JSON:
{json.dumps({"comments": batch}, ensure_ascii=False)}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLEANING_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        )

        raw_output = (response.choices[0].message.content or "").strip()

        try:
            cleaned_batch = json.loads(raw_output)
            if not isinstance(cleaned_batch, list):
                raise ValueError("Model did not return a JSON list")
            cleaned_batch = [str(x) for x in cleaned_batch]
        except Exception:
            # Fallback: if parsing fails, just re-use original comments for this batch
            cleaned_batch = batch

        cleaned_all.extend(cleaned_batch)

    # Safety: align length
    if len(cleaned_all) != len(comments):
        cleaned_all = cleaned_all[: len(comments)]

    return cleaned_all


def summarize_comments_with_llm(comments: List[str], max_sample: int = 80) -> str:
    """
    Use OpenAI chat completions to summarize a sample of cleaned comments.
    """
    if not comments:
        return "No comments available to summarize."

    sample = comments[:max_sample]
    joined = "\n".join(f"- {c}" for c in sample)

    prompt = f"""
Here is a sample of cleaned Reddit comments:

{joined}

Now write the summary as requested in the instructions.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUMMARY_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    return (response.choices[0].message.content or "").strip()


def clean_node(state: CommentPipelineState) -> dict:
    cleaned = clean_comments_with_llm(state["comments"])
    return {"cleaned_comments": cleaned}


def summarize_node(state: CommentPipelineState) -> dict:
    summary = summarize_comments_with_llm(state["cleaned_comments"])
    return {"summary": summary}


# Build LangGraph pipeline
graph = StateGraph(CommentPipelineState)
graph.add_node("clean_comments", clean_node)
graph.add_node("summarize", summarize_node)
graph.set_entry_point("clean_comments")
graph.add_edge("clean_comments", "summarize")
graph.add_edge("summarize", END)
pipeline_app = graph.compile()

# -----------------------------------------------------------------------------
# Reddit fetching via public API (no OAuth)
# -----------------------------------------------------------------------------
USER_AGENT = "Mozilla/5.0 (compatible; reddit-sentiment-agent/0.1; +https://example.com)"


def extract_comments_from_children(children: list, collected: List[str], max_comments: int):
    """
    Recursively traverse the Reddit JSON comment tree to collect comment bodies.
    """
    for child in children:
        kind = child.get("kind")
        data = child.get("data", {})

        if kind == "t1":  # comment
            body = (data.get("body") or "").strip()
            if body and body not in ("[deleted]", "[removed]"):
                collected.append(body.replace("\n", " "))
                if len(collected) >= max_comments:
                    return

            replies = data.get("replies")
            if isinstance(replies, dict):
                more_children = replies.get("data", {}).get("children", [])
                extract_comments_from_children(more_children, collected, max_comments)
                if len(collected) >= max_comments:
                    return

        # kind == "more" or others: skip


def fetch_post_and_comments(url: str, max_comments: int = 300):
    """
    Fetch a single Reddit post and its comments using the public API endpoint:
    https://api.reddit.com/comments/{post_id}
    """
    if "/comments/" not in url:
        raise ValueError("This does not look like a valid Reddit post URL (missing '/comments/').")

    # Extract post_id between /comments/ and the next /
    try:
        post_id = url.split("/comments/")[1].split("/")[0]
    except Exception:
        raise ValueError("Could not parse post ID from the given Reddit URL.")

    api_url = f"https://api.reddit.com/comments/{post_id}?limit={max_comments}&depth=10"

    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(api_url, headers=headers, timeout=15)

    if resp.status_code != 200:
        raise ValueError(f"Reddit API returned status {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        snippet = resp.text[:300]
        raise ValueError(
            f"Reddit did not return JSON. First 300 characters of response:\n{snippet}"
        ) from e

    # Expected shape: [ post_listing, comments_listing ]
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Unexpected Reddit API response format.")

    # Post metadata
    post_listing = data[0]
    post_children = post_listing.get("data", {}).get("children", [])
    if not post_children:
        raise ValueError("No post data found in Reddit response.")

    post_data = post_children[0].get("data", {})
    post_info = {
        "title": post_data.get("title", ""),
        "selftext": post_data.get("selftext", ""),
        "score": post_data.get("score", 0),
        "num_comments": post_data.get("num_comments", 0),
        "permalink": "https://www.reddit.com" + post_data.get("permalink", ""),
        "subreddit": post_data.get("subreddit", ""),
    }

    # Comments tree
    comments_listing = data[1]
    comments_root = comments_listing.get("data", {}).get("children", [])

    comments: List[str] = []
    extract_comments_from_children(comments_root, comments, max_comments)

    return post_info, comments


def search_reddit_posts(query: str, limit: int = 10):
    """
    Search Reddit posts for a given query using the public search endpoint.
    Returns a list of lightweight post dicts (id, title, subreddit, score, num_comments, permalink).
    """
    url = "https://api.reddit.com/search"
    params = {
        "q": query,
        "limit": limit,
        "sort": "relevance",
        "type": "link",
        "restrict_sr": False,
    }
    headers = {"User-Agent": USER_AGENT}

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"Reddit search returned status {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        snippet = resp.text[:300]
        raise ValueError(
            f"Reddit search did not return JSON. First 300 characters of response:\n{snippet}"
        ) from e

    children = data.get("data", {}).get("children", [])
    posts = []
    for child in children:
        d = child.get("data", {})
        posts.append(
            {
                "id": d.get("id"),
                "title": d.get("title", ""),
                "subreddit": d.get("subreddit", ""),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "permalink": "https://www.reddit.com" + d.get("permalink", ""),
            }
        )

    return posts

# -----------------------------------------------------------------------------
# Sentiment helper
# -----------------------------------------------------------------------------
def classify_sentiment(cleaned_text: str) -> str:
    """
    Use your VADER model to map compound score to Positive / Negative / Neutral.
    """
    scores = model.polarity_scores(cleaned_text)
    compound = scores["compound"]
    if compound > 0:
        return "Positive"
    elif compound < 0:
        return "Negative"
    else:
        return "Neutral"

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🧠 Reddit Comment Sentiment Analyzer (Agentic, Single / Multi Post)")
st.write(
    "Analyze Reddit sentiment either by **single post URL** or by **topic search** "
    "(top N posts). Comments are cleaned with an LLM agent (LangGraph) and "
    "sentiment is computed with VADER (`model_pickle`)."
)

mode = st.radio(
    "Choose input mode",
    ["Single post URL", "Topic search (multi-post)"],
    horizontal=True,
)

analysis_single_clicked = False
analysis_topic_clicked = False

if mode == "Single post URL":
    post_url = st.text_input(
        "Enter the Reddit post link here",
        placeholder="https://www.reddit.com/r/.../comments/POST_ID/...",
        key="single_post_url",
    )
    max_comments_single = st.slider(
        "Max comments to analyze (per post)",
        50,
        500,
        300,
        step=50,
        key="max_comments_single",
    )
    analysis_single_clicked = st.button("Analyze single post 🔍")

elif mode == "Topic search (multi-post)":
    topic_query = st.text_input(
        "Enter a topic / keyword to search on Reddit",
        placeholder="e.g. iphone battery life, generative AI, mental health",
        key="topic_query",
    )
    num_posts = st.slider(
        "Number of Reddit posts to analyze",
        1,
        10,
        5,
        step=1,
        key="num_posts",
    )
    max_comments_multi = st.slider(
        "Max comments per post",
        20,
        300,
        100,
        step=20,
        key="max_comments_multi",
    )
    analysis_topic_clicked = st.button("Search & analyze topic 🔍")

# -------------------------------------------------------------------------
# MODE 1: Single post pipeline
# -------------------------------------------------------------------------
if mode == "Single post URL" and analysis_single_clicked:
    if not post_url.strip():
        st.error("Please paste a Reddit post URL.")
        st.session_state["analysis"] = None
    else:
        # Step 1: Fetch post + comments
        with st.spinner("Fetching Reddit post and comments..."):
            try:
                post_info, comments = fetch_post_and_comments(
                    post_url, max_comments=max_comments_single
                )
            except Exception as e:
                st.error(f"Could not fetch comments from Reddit. Error: {e}")
                comments = []

        if not comments:
            st.warning("No comments found to analyze for this post.")
            st.session_state["analysis"] = None
        else:
            # Step 2: LangGraph pipeline (clean + summarize)
            with st.spinner("Running LangGraph pipeline: cleaning comments and summarizing..."):
                initial_state: CommentPipelineState = {
                    "comments": comments,
                    "cleaned_comments": [],
                    "summary": "",
                }
                final_state = pipeline_app.invoke(initial_state)

            cleaned_comments = final_state["cleaned_comments"]
            summary_text = final_state["summary"]

            # Step 3: Sentiment classification
            positive: List[str] = []
            negative: List[str] = []
            neutral: List[str] = []

            for original, cleaned in zip(comments, cleaned_comments):
                cleaned = (cleaned or "").strip()
                if not cleaned:
                    continue
                label = classify_sentiment(cleaned)
                if label == "Positive":
                    positive.append(original)
                elif label == "Negative":
                    negative.append(original)
                else:
                    neutral.append(original)

            sentiment_count_dict = {
                "Positive": [len(positive)],
                "Negative": [len(negative)],
                "Neutral": [len(neutral)],
            }

            st.session_state["analysis"] = {
                "mode": "single",
                "post_info": post_info,
                "topic_query": None,
                "search_posts": None,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "sentiment_counts": sentiment_count_dict,
                "summary": summary_text,
                "num_raw_comments": len(comments),
            }

# -------------------------------------------------------------------------
# MODE 2: Topic search (multi-post) pipeline
# -------------------------------------------------------------------------
if mode == "Topic search (multi-post)" and analysis_topic_clicked:
    if not topic_query.strip():
        st.error("Please enter a topic / keyword.")
        st.session_state["analysis"] = None
    else:
        # Step 0: Search Reddit posts
        with st.spinner(f"Searching Reddit for '{topic_query}'..."):
            try:
                search_posts = search_reddit_posts(topic_query, limit=num_posts)
            except Exception as e:
                st.error(f"Reddit search failed. Error: {e}")
                search_posts = []

        if not search_posts:
            st.warning("No posts found for this topic.")
            st.session_state["analysis"] = None
        else:
            # Fetch comments for each post
            all_comments: List[str] = []
            posts_info: List[dict] = []

            with st.spinner("Fetching comments from top posts..."):
                for p in search_posts:
                    try:
                        post_info, comments = fetch_post_and_comments(
                            p["permalink"], max_comments=max_comments_multi
                        )
                    except Exception:
                        # Skip bad posts quietly
                        continue

                    if comments:
                        posts_info.append(post_info)
                        all_comments.extend(comments)

            if not all_comments:
                st.warning("Could not fetch comments from any of the top posts.")
                st.session_state["analysis"] = None
            else:
                # Run LangGraph pipeline on ALL comments combined
                with st.spinner(
                    "Running LangGraph pipeline on combined comments (clean + summarize)..."
                ):
                    initial_state: CommentPipelineState = {
                        "comments": all_comments,
                        "cleaned_comments": [],
                        "summary": "",
                    }
                    final_state = pipeline_app.invoke(initial_state)

                cleaned_all = final_state["cleaned_comments"]
                summary_text = final_state["summary"]

                # Sentiment classification on combined comments
                positive: List[str] = []
                negative: List[str] = []
                neutral: List[str] = []

                for original, cleaned in zip(all_comments, cleaned_all):
                    cleaned = (cleaned or "").strip()
                    if not cleaned:
                        continue
                    label = classify_sentiment(cleaned)
                    if label == "Positive":
                        positive.append(original)
                    elif label == "Negative":
                        negative.append(original)
                    else:
                        neutral.append(original)

                sentiment_count_dict = {
                    "Positive": [len(positive)],
                    "Negative": [len(negative)],
                    "Neutral": [len(neutral)],
                }

                st.session_state["analysis"] = {
                    "mode": "topic",
                    "post_info": None,
                    "topic_query": topic_query,
                    "search_posts": search_posts,
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "sentiment_counts": sentiment_count_dict,
                    "summary": summary_text,
                    "num_raw_comments": len(all_comments),
                }

# -------------------------------------------------------------------------
# Display analysis if available for the current mode
# -------------------------------------------------------------------------
analysis = st.session_state.get("analysis")

if analysis and analysis.get("mode") == ("single" if mode == "Single post URL" else "topic"):
    positive = analysis["positive"]
    negative = analysis["negative"]
    neutral = analysis["neutral"]
    sentiment_count_dict = analysis["sentiment_counts"]
    summary_text = analysis["summary"]
    num_raw_comments = analysis["num_raw_comments"]

    if analysis["mode"] == "single":
        # Single-post display
        post_info = analysis["post_info"]

        st.header("Post Info")
        st.subheader(post_info["title"])
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("**Subreddit**")
            st.write(f"r/{post_info['subreddit']}")
        with col_p2:
            st.markdown("**Upvotes**")
            st.write(post_info["score"])
        with col_p3:
            st.markdown("**Total comments (raw)**")
            st.write(post_info["num_comments"])

        if post_info["selftext"]:
            with st.expander("Show post body"):
                st.write(post_info["selftext"])

        st.markdown(f"**Comments actually analyzed (after truncation):** {num_raw_comments}")

    else:
        # Topic (multi-post) display
        topic_query = analysis["topic_query"]
        search_posts = analysis["search_posts"]

        st.header(f"Topic analysis for: `{topic_query}`")
        st.markdown(
            f"Combined comments from **{len(search_posts)}** top Reddit posts "
            f"(up to the chosen comments-per-post limit)."
        )

        # Show which links were used
        st.subheader("Reddit posts used")
        posts_df = pd.DataFrame(
            [
                {
                    "Subreddit": p["subreddit"],
                    "Title": p["title"],
                    "Score": p["score"],
                    "Comments (total)": p["num_comments"],
                    "Link": p["permalink"],
                }
                for p in search_posts
            ]
        )
        st.dataframe(posts_df, use_container_width=True)

        st.markdown(f"**Total comments actually analyzed (all posts combined):** {num_raw_comments}")

    # Common part: sentiment counts, charts, summary, comment browser

    # Sentiment counts
    st.subheader("Sentiment analysis on cleaned comments")
    data = pd.DataFrame(sentiment_count_dict)
    st.write("### Sentiment counts")
    st.table(data)

    # Charts
    st.subheader("Sentiment distribution")

    categories = ["Positive", "Negative", "Neutral"]
    counts = [data["Positive"][0], data["Negative"][0], data["Neutral"][0]]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    ax1, ax2 = axs.flat

    # Bar chart
    ax1.bar(categories, counts)
    ax1.set_title("Sentiment Count (Bar)")
    ax1.set_ylabel("Number of Comments")

    # Pie chart
    ax2.pie(
        counts,
        labels=categories,
        autopct="%6.2f",
        startangle=90,
    )
    ax2.set_title("Sentiment Distribution (Pie)")

    st.pyplot(fig)

    # LLM summary
    st.subheader("LLM summary of the discussion")
    st.write(summary_text)

    # Comment browser
    st.subheader("Browse comments by sentiment")
    comment_option = st.selectbox(
        "Select the comments you want to read",
        ["Positive", "Negative", "Neutral"],
        index=0,
    )

    selected_list = (
        positive if comment_option == "Positive"
        else negative if comment_option == "Negative"
        else neutral
    )

    st.write(f"Showing **{len(selected_list)}** {comment_option.lower()} comments:")
    for c in selected_list:
        st.markdown("---")
        st.write(c)
