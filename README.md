# AI Agent System — Fireworks AI Hackathon (Track 1)

A **token-efficient, accuracy-first AI agent** for the Fireworks AI Challenge. Uses a 3-tier processing pipeline to minimize Fireworks API token consumption while maintaining high answer quality across all 8 task categories.

## Architecture

```
Prompt → [Tier 0: Keyword Classifier] → category
               ↓ (uncertain)
         [LLM Router via Fireworks]
               ↓
         [Tier 1: Local Processors]     ← ZERO tokens for math / sentiment / NER
               ↓ (complex tasks)
         [Tier 2: Fireworks API]        ← minimal tokens, compressed prompts
               ↓
         /output/results.json
```

| Tier | Method | Token Cost |
|------|--------|-----------|
| 0 | Regex keyword classifier | **0** |
| 1a | Deterministic Python math solver | **0** |
| 1b | TextBlob sentiment analysis | **0** |
| 1c | spaCy NER (en_core_web_sm) | **0** |
| 2 | Fireworks API (llama-v3p3-70b) | Minimal |

## Local Development

### Prerequisites
- Python 3.11+
- A Fireworks AI API key

### Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Create .env (never committed or baked into Docker image)
cat > .env << EOF
FIREWORKS_API_KEY=fw_your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/llama-v3p3-70b-instruct
EOF

# 3. Run agent
python agent.py

# 4. Check output
cat output/results.json
```

### Run Tests
```bash
pip install pytest
pytest tests/ -v
```

## Docker Deployment

### Build & Run (Recommended)
```bash
# Uses docker-compose.yml which reads .env automatically
docker compose up --build
```

### Manual Build (for linux/amd64 — required for harness)
```bash
# Build (always target linux/amd64)
docker buildx build --platform linux/amd64 -t agent-system:latest -f Dockerfile.uv .

# Run
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  --env-file .env \
  agent-system:latest
```

### Push to GitHub Container Registry
```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Tag
docker tag agent-system:latest ghcr.io/YOUR_GITHUB_USERNAME/agent-system:latest

# Push
docker push ghcr.io/YOUR_GITHUB_USERNAME/agent-system:latest
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|---------|
| `FIREWORKS_API_KEY` | Provided by harness — do **not** use your own | Yes |
| `FIREWORKS_BASE_URL` | All Fireworks calls routed through this | Yes |
| `ALLOWED_MODELS` | Comma-separated permitted model IDs | Yes |

> **Never hardcode these values or bundle `.env` in the Docker image.**

## Pre-Submission Checklist

- [ ] `output/results.json` contains `task_id` and `answer` keys (not `result`)
- [ ] All tasks are processed (no `[:3]` cap)
- [ ] No `GOOGLE_API_KEY` references in code
- [ ] Image built with `--platform linux/amd64`
- [ ] No `.env` file in image
- [ ] Image < 10 GB compressed
- [ ] Exit code 0 on success
- [ ] Local end-to-end test passes with all 8 practice tasks

## Scoring Strategy

- **Token efficiency rank**: fewer Fireworks tokens = higher rank
- Local processing (math/sentiment/NER) saves ~3–4 Fireworks API calls per evaluation run
- System prompts kept under 25 words to minimize input tokens
- Per-category `max_tokens` limits control output token spend
