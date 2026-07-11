FROM python:3.11-slim

WORKDIR /app

# Install system deps for spaCy and llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download TextBlob corpora
RUN python -c "import textblob; textblob.download_corpora('lite')" 2>/dev/null || \
    python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Download Local LLM model (Qwen2.5-1.5B-Instruct-GGUF)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='/app/models')"


COPY agent.py .

RUN mkdir -p /input /output

# Injected at runtime by harness
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL=""
ENV ALLOWED_MODELS=""

ENTRYPOINT ["python", "agent.py"]
