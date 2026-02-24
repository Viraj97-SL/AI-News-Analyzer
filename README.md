

# 🌐 AI News Analyzer & Publisher

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-FF4F00)
![Gemini API](https://img.shields.io/badge/AI-Google_Gemini-4285F4?logo=google&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Async_pg-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

An autonomous, multi-agent AI pipeline that acts as a fully automated AI research assistant and social media manager. Built with **LangGraph** and **FastAPI**, this system scrapes the latest AI/ML news, evaluates credibility, writes summaries using Google Gemini, generates cyberpunk-themed news cards, and waits for Human-in-the-Loop (HITL) email approval before publishing to LinkedIn and a newsletter.

---

## ✨ Key Features

- **Multi-Source Data Aggregation:** Parallel asynchronous scraping from ArXiv, RSS feeds (DeepMind, OpenAI, MIT Tech Review), Tavily Search, and Serper (Google News).
- **Intelligent Deduplication & Scoring:** Automatically merges duplicate stories across sources and scores credibility based on domain reputation.
- **Tiered LLM Routing:** Uses cost-effective models (Gemini Flash-Lite) for topic classification and high-reasoning models (Gemini Flash/Pro) for complex summarization and drafting.
- **Human-in-the-Loop (HITL):** Pauses graph execution using LangGraph's `interrupt()` to send an admin email via Resend with secure, HMAC-signed JWT tokens for one-click **Approve** or **Reject** actions.
- **Dynamic Image Generation:** Uses a headless Chromium browser (`html2image`) to render beautiful, cyberpunk/glassmorphism 1200x627px news cards using Jinja2 templates and CSS.
- **Production-Ready:** Containerized with Docker, asynchronous database operations (SQLAlchemy + Asyncpg), and fully deployable to cloud platforms like Railway.

---

## 🏗️ Pipeline Architecture

The application is orchestrated as a directed acyclic graph (DAG) using **LangGraph**:

1. **`scrape_node`**: Fetches raw articles concurrently from diverse sources.
2. **`summarize_node`**: Merges, deduplicates, and uses Gemini to categorize and summarize the most relevant AI stories.
3. **`linkedin_gen_node`**: Acts as a social media strategist, drafting an engaging LinkedIn post with a hook and takeaways.
4. **`image_gen_node`**: Generates dark-mode, neon-accented graphical cards for the articles.
5. **`approval_node`**: Suspends the state machine and emails the preview to a human editor.
6. **`publish_node` / `revise_node`**: Based on the secure URL clicked in the email, either publishes the content or routes back to the LLM with human feedback for revision.

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

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

```

### 2. Environment Variables

Create a `.env` file in the root directory. Copy the contents from `.env.example` and fill in your API keys:

```env
# Core Database
DATABASE_URL=sqlite+aiosqlite:///./test.db  # Use PostgreSQL for production
APP_ENV=development

# Security
API_KEY=your-secret-trigger-key
APP_SECRET_KEY=generate-a-random-string
JWT_SECRET=generate-a-random-string

# AI & Search
GOOGLE_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key

# Email (Resend)
RESEND_API_KEY=your-resend-key
EMAIL_FROM=onboarding@resend.dev
EMAIL_TO=your.email@example.com

```

### 3. Run the API Server

```bash
uvicorn app.main:app --reload

```

You can now access the interactive API documentation at `http://localhost:8000/docs`.

### 4. Trigger a Pipeline Run

You can trigger the multi-agent pipeline using curl:

```bash
curl -X POST http://localhost:8000/api/v1/runs/trigger \
  -H "X-API-Key: your-secret-trigger-key"

```

---

## 🐳 Docker Deployment

The project includes a multi-stage Dockerfile optimized for production, which automatically installs the necessary Chromium dependencies for image generation.

```bash
# Build the image
docker compose build

# Run the database and web server
docker compose up -d

```

---

## ⏱️ Automated Scheduling (Cron)

For production, you can run the pipeline automatically (e.g., every Tuesday and Thursday at 9 AM UTC) using the provided trigger script.

If deploying on a platform like Railway:

1. Deploy the exact same repository as a secondary service.
2. Set the service type to **Cron Job**.
3. Set the schedule to `0 9 * * 2,4`.
4. Set the Start Command to:

```bash
PYTHONPATH=/app python cron/trigger.py

```

---

## 🛡️ License & Acknowledgements

* Built using [LangGraph](https://python.langchain.com/docs/langgraph/) for stateful, multi-actor LLM applications.
* UI inspiration for image generation features Dark Mode / Glassmorphism aesthetics.

