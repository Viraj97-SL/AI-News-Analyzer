

# 🌐 AI News & Research Analyst

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-FF4F00)
![Gemini API](https://img.shields.io/badge/AI-Google_Gemini-4285F4?logo=google&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Async_pg-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

An autonomous, dual-pipeline AI media empire. Built with **LangGraph** and **FastAPI**, this system operates two independent multi-agent workflows:
1. **The News Pipeline:** A high-volume AI news aggregator and summarizer.
2. **The Research Pipeline:** A deep-tech academic analyst that isolates and breaks down complex ArXiv papers.

Both pipelines feature Human-in-the-Loop (HITL) email approvals, dynamic cyberpunk-themed image generation, and automated social media publishing.

---

## ✨ Key Features

- **Dual-Brain Architecture:** Completely isolated state machines for News (`graph.py`) and Research (`research_graph.py`) to prevent cross-contamination of prompts and context.
- **Agentic Paper Selection:** Uses Gemini 2.5 Flash to rapidly scan 50+ ArXiv abstracts and select the single most groundbreaking paper of the week.
- **Deep Thematic Analysis:** Uses Gemini 2.5 Pro to act as a Principal Investigator, extracting the Core Problem, Methodology, Breakthroughs, and Limitations of dense academic papers.
- **Dynamic Cyberpunk Visuals:** Uses a headless Chromium browser (`html2image`) with Jinja2 and CSS to generate aesthetic, data-rich graphical cards for both standard news and deep research grid layouts.
- **Human-in-the-Loop (HITL):** Pauses graph execution via LangGraph's `interrupt()` to send an admin email via Resend with secure, HMAC-signed JWT tokens for one-click **Approve** or **Reject** publishing.
- **Future-Proofed for PaperBanana:** Built-in hooks to integrate cutting-edge agentic flowchart generation (PaperBanana) with automatic fallbacks to HTML-rendered visuals.

---

### 🏗️ Pipeline Architectures

````
┌─────────────────────────────────────────────────────────┐
│               Railway Cloud Environment                 │
│  [ FastAPI Web ]   [ News Cron ]   [ Research Cron ]    │
└───────────────────────────┬─────────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────┐                   ┌──────────────────┐
│ LangGraph Agent │                   │ LangGraph Agent  │
│ (News Pipeline) │                   │(Research Analyst)│
└────────┬────────┘                   └────────┬─────────┘
         │                                     │
         ├─ Scraper Tools                      ├─ ArXiv Specialist
         │  ├─ Tavily Search API               │  └─ Queries cs.AI/cs.LG
         │  ├─ Serper (Google News)            │
         │  ├─ RSS Aggregator                  ├─ Paper Selection Agent
         │  └─ ArXiv API                       │  └─ Gemini 2.5 Flash
         │                                     │
         ├─ Processing & NLP                   ├─ Deep Tech Analyst
         │  ├─ Hash Deduplicator               │  ├─ Gemini 2.5 Pro
         │  └─ Gemini Summarizer               │  └─ Extracts Methodology
         │                                     │
         ├─ Media Generation                   ├─ Media Generation
         │  ├─ LinkedIn Hook Gen               │  ├─ Cyberpunk Grid Card
         │  └─ html2image News Card            │  └─ (PaperBanana Hook)
         │                                     │
         └─ HITL Approval Node                 └─ HITL Approval Node
                       │                       │
                       └───────────┬───────────┘
                                   ▼
                 ┌───────────────────────────────────┐
                 │    Publishing & Distribution      │
                 │  ├─ Resend API (Approval Emails)  │
                 │  └─ LinkedIn API (Post & Image)   │
                 └───────────────────────────────────┘
         
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL (Railway)                     │
│  agent_runs | summaries | news_articles | lg_checkpoints    │
└─────────────────────────────────────────────────────────────┘
````
### 1. 📰 The News Pipeline (Runs Tuesdays)
1. **Fan-Out Scraping:** Concurrently scrapes Tavily Search, Serper, ArXiv, and RSS feeds.
2. **Deduplication & Scoring:** Merges duplicate stories and scores credibility based on domain reputation.
3. **Summarization:** Gemini generates cost-effective categorized summaries.
4. **Content & Image Gen:** Drafts a LinkedIn hook and renders a standard News Card.
5. **Approval:** Halts state and awaits human email approval to publish.

### 2. 🔬 The Research Pipeline (Runs Thursdays)
1. **Narrow Scraping:** Exclusively pulls from ArXiv (`cs.AI`, `cs.LG`, etc.).
2. **Select Best Paper:** Evaluates novelty and impact to isolate *one* primary paper.
3. **Deep Analysis:** Generates a highly technical breakdown (Problem, Methodology, Limitations).
4. **Visual Diagramming:** Generates a highly detailed Research Diagnostic grid card.
5. **Approval:** Halts state and awaits human email approval to publish the deep dive.

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.11+
- PostgreSQL (or SQLite for local testing)
- Chrome/Chromium (required for image generation)

### 1. Clone & Install
```bash
git clone [https://github.com/YourUsername/ai-news-analyzer.git](https://github.com/YourUsername/ai-news-analyzer.git)
cd ai-news-analyzer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
DATABASE_URL=sqlite+aiosqlite:///./test.db 
APP_ENV=development
API_KEY=your-secret-trigger-key
APP_SECRET_KEY=generate-a-random-string
JWT_SECRET=generate-a-random-string
GOOGLE_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
RESEND_API_KEY=your-resend-key
EMAIL_FROM=onboarding@resend.dev
EMAIL_TO=your.email@example.com

```

### 3. Run the API Server

```bash
uvicorn app.main:app --reload

```

Interactive API documentation available at `http://localhost:8000/docs`.

---

## ⏱️ Production Deployment (Railway Cron Jobs)

This system is designed to be deployed as a microservice cluster on Railway.

**Service 1: The Web API (24/7)**

* **Builder:** Dockerfile
* **Command:** Uses default Dockerfile command (`gunicorn`).

**Service 2: The News Aggregator (Tuesdays)**

* **Builder:** Dockerfile
* **Type:** Cron Job
* **Schedule:** `0 9 * * 2` (Tuesday 9AM UTC)
* **Command:** `python cron/trigger.py`
* **Variables:** Ensure `PYTHONPATH=/app` is set.

**Service 3: The Deep Tech Analyst (Thursdays)**

* **Builder:** Dockerfile
* **Type:** Cron Job
* **Schedule:** `0 9 * * 4` (Thursday 9AM UTC)
* **Command:** `python cron/research_trigger.py`
* **Variables:** Ensure `PYTHONPATH=/app` is set.

---

## 🛡️ Built With

* [LangGraph](https://python.langchain.com/docs/langgraph/) - Stateful orchestration.
* [FastAPI](https://fastapi.tiangolo.com/) - High-performance async web framework.
* [Google Gemini](https://ai.google.dev/) - Multimodal reasoning and summarization.
### VIRAJ BULUGAHAPITIYA - AI/ML ENGINEER
```

