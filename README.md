# Decoding-Reddit-Trends-with-Agentic-AI-Real-Time-Sentiment-Summarization
## Table of Contents
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [How it Works](#how-it-works)
- [Technical Stack](#technical-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Roadmap](#roadmap)

## About the Project

The Reddit Sentiment Agent is an intelligent, agentic-AI application designed to provide real-time, actionable insights from Reddit discussions. It transforms raw, often noisy social media data into structured sentiment analysis and concise summaries, helping users quickly grasp the pulse of any given topic or post without endless scrolling.

Leveraging an agentic approach powered by LangGraph, this project goes beyond simple generative AI. It implements a structured pipeline where an AI agent first intelligently cleans and normalizes the data, significantly improving the accuracy and relevance of subsequent sentiment scoring and summarization.

## Key Features

* **Dual-Mode Analysis:** Analyze a single viral Reddit thread or perform a broader topic search across multiple top-ranking posts (e.g., "iPhone battery life").
* **LLM-Powered Cleaning Agent:** Utilizes advanced LLMs (e.g., GPT-4o-mini) as a preprocessing agent to intelligently clean raw Reddit comment text. This includes handling slang, sarcasm, markdown artifacts, and normalization, preserving context while preparing data for accurate analysis.
* **Hybrid Sentiment Scoring:** Combines the contextual reasoning capabilities of LLMs for nuanced summarization with the speed and deterministic nature of VADER for consistent sentiment polarity scoring.
* **Instant "TL;DR" Summaries:** Generates concise, executive summaries of prevailing topics, emotions, and key discussion points within the analyzed Reddit data.
* **Public API Access:** Dynamically fetches live Reddit data, making it accessible without complex Reddit API authentication setups.
* **Intuitive User Interface:** A clean, interactive dashboard built with Streamlit for easy exploration of insights.

## How it Works

The core of the Reddit Sentiment Agent is its LangGraph-orchestrated pipeline, functioning as an "Intelligence Funnel":

1.  **The Chaos (Data Ingestion):** The application fetches raw comment data from Reddit using public APIs. This initial data is often messy, containing noise, varying language styles, and platform-specific markdown.
2.  **The Agent (Cleaning & Reasoning):** An LLM-powered agent takes this raw data and intelligently cleans it. This involves:
    * Removing non-essential markdown and special characters.
    * Normalizing language (e.g., expanding contractions).
    * Attempting to handle sarcasm or context-dependent phrases (to the extent possible with LLMs).
    * Preparing the text for reliable sentiment analysis.
3.  **Structured Insights & Sentiment (Analysis & Output):**
    * The cleaned text is then fed into a sentiment analysis module (VADER) to determine polarity (positive, negative, neutral).
    * An LLM summarizes the key themes, overall sentiment, and interesting discussion points from the cleaned and scored data.
    * Results are presented in an easy-to-understand Streamlit dashboard with visualizations (e.g., sentiment distribution charts) and a "TL;DR" summary.

## Technical Stack

* **Python:** The primary programming language.
* **Streamlit:** For building the interactive web user interface.
* **LangGraph:** To orchestrate the agentic workflow and state management between different AI agents.
* **OpenAI (GPT-4o-mini):** Utilized for the LLM-powered cleaning agent and for generating concise summaries.
* **NLTK (VADER):** For robust, rule-based sentiment intensity analysis on cleaned text.
* **Matplotlib / Pandas:** For data processing and visualization within the Streamlit application.
* **Reddit APIs:** For fetching live public Reddit data.

## Getting Started

Follow these steps to get your local copy up and running.

### Prerequisites

* Python 3.8+
* An OpenAI API Key (for GPT-4o-mini)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/amey-ghate/Decoding-Reddit-Trends-with-Agentic-AI-Real-Time-Sentiment-Summarization.git]
    cd reddit-sentiment-agent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Set up your OpenAI API Key:**
    Create a `.env` file in the root of your project directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *Make sure to never hardcode API keys directly into your code.*

### Running the Application

1.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser (usually `http://localhost:8501`).

## Usage

1.  **Navigate to the Streamlit application** in your browser.
2.  **Choose your analysis mode:**
    * **Single Reddit URL:** Paste the URL of a specific Reddit post you want to analyze.
    * **Topic Search:** Enter a keyword or phrase, and the agent will fetch relevant top posts to analyze collectively.
3.  **Click "Analyze":** The agentic pipeline will execute, cleaning the data, performing sentiment analysis, and generating a summary.
4.  **View Results:** Explore the sentiment distribution charts, the "TL;DR" summary, and potentially a breakdown of key themes.

## Roadmap

* **Enhanced LLM Cleaning:** Experiment with more sophisticated prompt engineering for cleaning to handle increasingly complex linguistic nuances (e.g., subtle sarcasm detection).
* **Named Entity Recognition (NER):** Identify key entities (people, organizations, products) discussed within the comments.
* **Temporal Analysis:** Track sentiment changes over time for long-running discussions.
* **User Filtering:** Option to filter comments by Reddit user or karma score.
* **Export Options:** Allow users to export results as CSV or JSON.
* **More LLM Options:** Integrate support for other LLMs (e.g., Llama 3 via Groq) for varied performance and cost.
