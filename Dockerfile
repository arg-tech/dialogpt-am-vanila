FROM python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install amf-fast-inference without its deps (pinned openvino==2024.0.0 not available on this platform)
RUN pip install --no-cache-dir --no-deps amf-fast-inference==0.0.3

# Preload the Hugging Face model and save it to /app/model
RUN python -c "from transformers import AutoModelForSequenceClassification, GPT2Tokenizer; \
    model_name = 'debela-arg/dialogpt-am-medium-context'; \
    model = AutoModelForSequenceClassification.from_pretrained(model_name); \
    tokenizer = GPT2Tokenizer.from_pretrained(model_name); \
    model.save_pretrained('/app/model'); \
    tokenizer.save_pretrained('/app/model')"

COPY . .

EXPOSE 5015

CMD ["gunicorn", "--bind", "0.0.0.0:5015", "--workers", "1", "--timeout", "120", "main:app"]
