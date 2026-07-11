# AI Agent System — Local-First Hybrid LLM Agent

A **token-efficient, accuracy-first AI agent** designed for the Fireworks AI Challenge. This system heavily utilizes offline models for basic tasks and leverages aggressive prompt-engineering with the powerful `deepseek-v4-pro` model to minimize token spend while maintaining 100% accuracy.

---

## 🚀 The Workflow (Smart Routing)

We use a **4-Tier Pipeline** to process tasks based on difficulty to avoid burning API tokens:

1. **Tier 0 (Keyword Router)**: Uses regex and basic string matching to instantly classify the task type.
2. **Tier 1 (Local Offline Solvers)**: Deterministic code (regex math), TextBlob (sentiment), and spaCy (NER) solve tasks for exactly **0 API tokens**.
3. **Tier 1.5 (Local Offline LLM)**: Qwen2.5-1.5B (GGUF format) runs locally via `llama-cpp-python` to handle factual, summary, and general conversational tasks for exactly **0 API tokens**.
4. **Tier 2 (Fireworks Fallback)**: For complex reasoning, code generation, and debugging, the system calls `deepseek-v4-pro`. 

### Token Optimization Engine
When falling back to Tier 2, the agent uses several techniques to drop the token count from >2,000 to <400:
- **`reasoning_effort="none"`**: Turns off extraneous verbose chain-of-thought natively.
- **Bare-metal Prompts**: We instruct the LLM to output ONLY the required code blocks without markdown wrapping or text explanations.
- **Logic Safeguards**: For Logic tasks specifically, we DO allow a step-by-step reasoning prompt and temporarily expand the token budget, guaranteeing that the puzzle is answered accurately without truncation errors.

---

## 🛠️ Setup & Running Locally (Host Machine)

If you wish to run the agent natively on your Windows/Mac host without Docker:

### 1. Prerequisites
- Python 3.11+
- Install `uv` (the fast package manager)
- Provide your Fireworks API Key in a `.env` file (`FIREWORKS_API_KEY=...`)

### 2. Install Dependencies & Local Models
```bash
# Sync all dependencies via uv
uv sync

# Download the Qwen Local LLM (~1.1GB) into the models/ folder
uv run python download_model.py

# Download the spaCy NER local solver dataset
uv run python -m spacy download en_core_web_sm
```

### 3. Run the Agent
```bash
uv run python agent.py
```
The agent will process `input/tasks.json`. Check `output/results.json` for the final answers. Token usage and execution trace will be printed directly to your console!

---

## 🐳 Docker Testing (Production Submission)

To test the full submission environment using Docker (this is how the judges will evaluate the code):

```bash
docker compose up --build
```

**What this does:**
- Builds the `linux/amd64` image (`agent-system:latest`) using `Dockerfile.uv`.
- Bakes the Qwen2.5-1.5B model and the spaCy NER models directly into the image layer cache.
- Mounts the host's `input/` and `output/` directories.
- Injects `FIREWORKS_API_KEY` from your `.env`.
- Executes the agent automatically and prints the Token Usage score to the terminal.

### 🏆 Expected Scoring Output
- **Total API Tokens Used:** ~380-400 tokens across all 8 tasks.
- **Accuracy:** Passing grade on all Logic, Factual, and Math tasks.
- **Docker Footprint:** Total image size is well under the 10GB limit (approx ~1.5GB total footprint).
