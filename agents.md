AI Agent System Documentation
Overview
This document outlines the architecture, development workflow, and deployment strategy for the General-Purpose AI Agent built for the Fireworks AI Challenge. The system uses a supervisor-worker pattern with specialized prompts to achieve high accuracy while minimizing token usage.

Architecture Goals
Primary Objectives
Token Efficiency: Minimize API token consumption while maintaining accuracy

Accuracy: Pass the LLM-Judge threshold across all 8 capability categories

Runtime: Complete all tasks within the 10-minute limit

Reliability: Graceful fallback mechanisms for edge cases

Design Principles
Single-Pass Processing: Each task flows through supervisor -> worker -> result (no loops)

Specialized Prompts: Each worker has a focused, minimal prompt for its category

Deterministic Outputs: Low temperature (0.1) for consistent, predictable results

Structured Validation: Pydantic models for input/output validation

Graceful Degradation: Fallback LLM call if main workflow fails

System Architecture
Core Components
text
+-------------------------------------------------------------+
|                      AGENT SYSTEM                            |
+-------------------------------------------------------------+
|                                                             |
|  +-------------+    +--------------+    +-------------+     |
|  |   Input     |    |  Supervisor  |    |   Workers   |     |
|  |  /input/    |--->|  (Router)    |--->|  (Special-  |     |
|  |  tasks.json |    |              |    |   ized)     |     |
|  +-------------+    +--------------+    +-------------+     |
|                                                             |
|  +-----------------------------------------------------+   |
|  |             Output /output/results.json              |   |
|  +-----------------------------------------------------+   |
|                                                             |
+-------------------------------------------------------------+
Component Details
1. Supervisor (Router)
Function: Classifies task category using a lightweight LLM call

Output: Category name (factual, math, sentiment, etc.)

Optimization: Uses with_structured_output() for clean parsing

Token Cost: ~50-100 tokens per task

2. Workers (Specialized Processors)
Function: Process tasks in their specific domain

Implementation: 8 specialized prompts, one per category

Pattern: Factory function creates workers with category-specific prompts

Token Cost: ~100-200 tokens per task (system prompt + response)

3. State Management
Structure: TypedDict with minimal state fields

Fields: messages, task_id, original_prompt, category, worker_output

Purpose: Lightweight state passing between nodes

4. Fallback Handler
Trigger: When main workflow fails

Action: Direct LLM call with general-purpose prompt

Purpose: Ensures graceful degradation

Data Flow
Input Reading: Load tasks from /input/tasks.json

Task Processing: For each task:

Create initial state

Run supervisor to detect category

Route to appropriate worker

Execute worker with specialized prompt

Extract answer

Output Writing: Write results to /output/results.json

Specialized Prompts
Category	Prompt Focus	Token Budget
factual	Explain concepts concisely, under 150 words	~50 tokens
math	Show only essential steps, final answer clear	~50 tokens
sentiment	JSON output, brief justification	~40 tokens
summary	Follow exact length constraints	~40 tokens
ner	JSON output with entity types	~40 tokens
code_debug	Brief bug explanation + corrected code	~60 tokens
logic	List constraints, concise reasoning	~50 tokens
code_gen	Clean code with type hints	~60 tokens
general	Direct, factual, under 200 words	~30 tokens
Development Setup
Prerequisites
Python 3.11+

UV (fast Python package installer)

Docker (for containerization)

Fireworks AI API key

Local Development with UV
Install UV

bash
curl -LsSf https://astral.sh/uv/install.sh | sh
Initialize Project

bash
uv init agent-system
cd agent-system
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install Dependencies

bash
uv add langgraph langchain-openai langchain-core pydantic python-dotenv
uv add --dev pytest black mypy
Create Environment File

bash
# .env
FIREWORKS_API_KEY=your_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/llama-v3p3-70b-instruct
Run Locally

bash
# Create input directory with test tasks
mkdir -p input output
echo '[{"task_id": "t1", "prompt": "Explain photosynthesis"}]' > input/tasks.json

# Run agent
python agent.py
Docker Deployment
Build Image
bash
# Build with UV in Docker
docker build -t agent-system:latest -f Dockerfile.uv .

# Or build with pip
docker build -t agent-system:latest -f Dockerfile .
Run Container
bash
# Basic run
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL=$FIREWORKS_BASE_URL \
  -e ALLOWED_MODELS=$ALLOWED_MODELS \
  agent-system:latest
Push to Registry
bash
# Tag image
docker tag agent-system:latest ghcr.io/your-username/agent-system:latest

# Login to registry
echo $GITHUB_TOKEN | docker login ghcr.io -u your-username --password-stdin

# Push image
docker push ghcr.io/your-username/agent-system:latest
Dockerfile Options
Option 1: Fast with UV (Recommended)
dockerfile
# Dockerfile.uv
FROM python:3.11-slim

# Install UV
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with UV
RUN uv sync --frozen --no-dev

# Copy application code
COPY agent.py .

# Create directories
RUN mkdir -p /input /output

# Environment variables (overridden at runtime)
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL=""
ENV ALLOWED_MODELS=""

ENTRYPOINT ["uv", "run", "python", "agent.py"]
Option 2: Traditional with Pip
dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent.py .

# Create directories
RUN mkdir -p /input /output

# Environment variables
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL=""
ENV ALLOWED_MODELS=""

ENTRYPOINT ["python", "agent.py"]
Project Structure
text
agent-system/
├── agent.py              # Main application code
├── pyproject.toml        # UV project configuration
├── uv.lock              # UV lock file (auto-generated)
├── requirements.txt      # Pip dependencies (if using pip)
├── Dockerfile           # Docker build file
├── Dockerfile.uv        # Docker build file (UV version)
├── .env                 # Local environment variables (gitignored)
├── .gitignore           # Git ignore file
├── input/               # Input directory (mounted volume)
│   └── tasks.json       # Task definitions
├── output/              # Output directory (mounted volume)
│   └── results.json     # Results file
└── tests/               # Test directory
    ├── test_agent.py    # Unit tests
    └── fixtures/        # Test fixtures
        └── tasks.json   # Sample tasks for testing
Configuration
Environment Variables
Variable	Description	Required
FIREWORKS_API_KEY	API key provided by harness	Yes
FIREWORKS_BASE_URL	Base URL for API calls	Yes
ALLOWED_MODELS	Comma-separated list of model IDs	Yes
Runtime Settings
python
# agent.py configuration section
MODEL = ALLOWED_MODELS[0]        # Use first allowed model
TEMPERATURE = 0.1                # Low for deterministic outputs
MAX_TOKENS = 2048                # Max response length
INPUT_FILE = "/input/tasks.json"
OUTPUT_FILE = "/output/results.json"
Testing
Unit Tests
python
# tests/test_agent.py
import pytest
from agent import TaskProcessor, detect_task_category

def test_category_detection():
    assert detect_task_category("Calculate 15% of 200") == "math"
    assert detect_task_category("Who is the CEO of Apple?") == "factual"
    assert detect_task_category("This product is amazing!") == "sentiment"

def test_task_processor():
    processor = TaskProcessor()
    result = processor.process_task("t1", "Explain gravity")
    assert result["task_id"] == "t1"
    assert result["success"] == True
    assert len(result["answer"]) > 0

def test_output_format():
    # Test that output matches expected JSON schema
    import json
    with open("input/tasks.json", "r") as f:
        tasks = json.load(f)
    
    # Process and validate
    from agent import main
    # ... validation logic
Running Tests
bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=agent tests/

# Run specific test
pytest tests/test_agent.py::test_category_detection
Deployment Checklist
Before Submission
Test with sample tasks locally

Verify all environment variables are read correctly

Ensure Docker image builds successfully

Check image size < 10GB

Test input/output directory mounting

Verify JSON output format

Confirm Fireworks API calls use FIREWORKS_BASE_URL

No hardcoded API keys or model IDs

Exit code 0 on success

Runtime under 10 minutes

Docker Image Tags
bash
# Version tags
docker tag agent-system:latest ghcr.io/username/agent-system:latest
docker tag agent-system:latest ghcr.io/username/agent-system:v1.0.0

# Development tags
docker tag agent-system:latest ghcr.io/username/agent-system:dev
Performance Monitoring
Token Usage Tracking
python
# Add token tracking to agent.py
import logging

def process_task_with_tracking(self, task_id: str, prompt: str):
    # Track token usage
    logging.info(f"Processing task {task_id}")
    start_time = time.time()
    
    result = self.process_task(task_id, prompt)
    
    elapsed = time.time() - start_time
    logging.info(f"Task {task_id} completed in {elapsed:.2f}s")
    
    return result
Metrics to Monitor
Token per Task: Average tokens consumed per task

Success Rate: Percentage of tasks processed successfully

Runtime per Task: Time taken per task

Fallback Rate: Frequency of fallback handler usage

Security Guidelines
Never hardcode API keys - Use environment variables

No .env in Docker image - Variables injected at runtime

Validate inputs - Sanitize prompt content before processing

Error logging - Log errors but don't expose sensitive data

Rate limiting - Handle API rate limits gracefully

Troubleshooting
Common Issues
Issue	Solution
Docker image too large	Use slim base image, remove cache
Token usage too high	Reduce prompt length, lower temperature
Invalid JSON output	Use structured output, validate response
Timeout errors	Add retry logic, reduce complexity
API rate limiting	Implement exponential backoff
Debug Commands
bash
# Check Docker image size
docker images | grep agent-system

# View container logs
docker logs <container_id>

# Run with bash for debugging
docker run --rm -it agent-system:latest /bin/bash

# Test locally with environment
python -c "import os; print(os.environ.get('FIREWORKS_API_KEY'))"
References
LangGraph Documentation: https://langchain-ai.github.io/langgraph/

Fireworks AI API: https://docs.fireworks.ai/

Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/

UV Package Manager: https://docs.astral.sh/uv/

Notes
Token Efficiency is Key: The leaderboard is ranked by token usage

Accuracy First: Must pass threshold before token ranking applies

Single Submission: One Docker image per team

Rate Limit: 10 submissions per hour per team

Last Updated: 2026-07-07
Version: 1.0.0
Maintainer: AI Engineering Team




