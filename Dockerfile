FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

RUN mkdir -p /input /output

ENV GOOGLE_API_KEY=""
ENV FIREWORKS_API_KEY=""
ENV FIREWORKS_BASE_URL=""
ENV ALLOWED_MODELS=""

ENTRYPOINT ["python", "agent.py"]
