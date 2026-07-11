FROM python:3.11-slim

WORKDIR /app

# Install system deps for spaCy (if needed for C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download TextBlob corpora
RUN python -c "import textblob; textblob.download_corpora('lite')" 2>/dev/null || \
    python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

COPY agent.py .

RUN mkdir -p /input /output

# Injected at runtime by harness
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL=""
ENV ALLOWED_MODELS=""

ENTRYPOINT ["python", "agent.py"]
