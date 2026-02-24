# 🤖 AI News Summarizer

**A LangGraph multi-agent system that autonomously scrapes, analyses, summarises, and publishes AI/ML news on a Tuesday/Thursday schedule — with human-in-the-loop approval for LinkedIn publishing.**

## Architecture

```
START → Scrape (4 sources in parallel) → Merge → Deduplicate → Credibility Score
  → Analyse → Summarise → Generate LinkedIn Post → Generate Image Cards
  → Human Approval [interrupt] → Email Newsletter + LinkedIn Publish
```

**Tech Stack**: LangGraph · FastAPI · Gemini (Flash-Lite / Flash / Pro) · PostgreSQL · Resend · Railway

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/ai-news-summarizer.git
cd ai-news-summarizer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (at minimum: GOOGLE_API_KEY, TAVILY_API_KEY)
```

### 3. Run locally

```bash
# Option A: Direct (uses SQLite)
uvicorn app.main:app --reload

# Option B: Docker (uses PostgreSQL)
docker compose up
```

### 4. Trigger a pipeline run

```bash
curl -X POST http://localhost:8000/api/v1/runs/trigger \
  -H "X-API-Key: your-api-key"
```

## Development

```bash
# Lint & format
ruff check . && ruff format .

# Run tests
pytest tests/unit -v

# Type check
mypy app/ --ignore-missing-imports

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Deployment (Railway)

1. Push to GitHub
2. Connect repo to Railway
3. Add PostgreSQL addon
4. Set environment variables in Railway dashboard
5. Create a cron service with schedule `0 9 * * 2,4`

CI/CD runs automatically on push to `main` via GitHub Actions.

## Project Structure

```
app/
├── agents/          # LangGraph supervisor graph + agent nodes
├── api/v1/routes/   # FastAPI endpoints
├── core/            # Config, security, logging
├── models/          # SQLAlchemy ORM
├── schemas/         # Pydantic request/response
├── services/        # Email, LinkedIn, image generation
└── templates/       # MJML email + HTML image card templates
```

## Estimated Monthly Cost

| Service        | Cost       |
|---------------|------------|
| Railway Hobby  | $5         |
| PostgreSQL     | ~$1        |
| Gemini API     | ~$25–50    |
| Tavily         | ~$30       |
| Resend         | Free       |
| **Total**      | **$35–75** |

## License

MIT
