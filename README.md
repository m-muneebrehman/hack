# AI Agent System

A high-performance, token-efficient AI agent designed for the Fireworks AI Challenge. This system uses a hybrid supervisor-worker pattern to process tasks with high accuracy while drastically minimizing API costs.

## Architecture Workflow

1. **Input Parsing**: Tasks are loaded sequentially from `/input/tasks.json`.
2. **Zero-Token Router**: A heuristic keyword classifier attempts to map the prompt to one of 8 specialized categories. This bypasses the LLM entirely for obvious tasks, saving ~50-100 tokens per request.
3. **LLM Supervisor (Fallback)**: If the keyword classifier is uncertain, a lightweight LLM router categorizes the prompt using strict structured output validation (`json_schema`).
4. **Specialized Workers**: The task is sent to the LLM accompanied by a hyper-condensed system prompt specific to its category, ensuring accurate and token-efficient responses.
5. **Output Generation**: Processed results and error fallbacks are logged to `agent.log` and the final payload is written to `/output/results.json`.

---

## Local Development

We use `uv` for ultra-fast Python dependency management.

### 1. Setup
Create a `.env` file in the root directory and add your API key:
```env
GOOGLE_API_KEY="your_api_key_here"
```

### 2. Run
Execute the agent directly:
```bash
uv run python agent.py
```
*Note: Test tasks are located in `input/tasks.json`. Check `agent.log` for token usage and routing decisions.*

---

## Docker Deployment

To run the agent in a secure, containerized environment matching the challenge requirements:

### 1. Build the Image
We provide an optimized `Dockerfile.uv` for faster builds:
```bash
docker build -t agent-system:latest -f Dockerfile.uv .
```

### 2. Run the Container
You must mount the `input` and `output` directories so the container can read your tasks and save the results. Inject your API key as an environment variable.

**Windows (PowerShell):**
```powershell
docker run --rm `
  -v "${PWD}/input:/input" `
  -v "${PWD}/output:/output" `
  -e GOOGLE_API_KEY="your_api_key_here" `
  agent-system:latest
```

**Linux / macOS (Bash):**
```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e GOOGLE_API_KEY="your_api_key_here" \
  agent-system:latest
```
