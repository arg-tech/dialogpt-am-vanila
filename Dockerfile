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

RUN pip install --no-cache-dir torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Download from HF and export to OpenVINO IR with INT8 quantisation at build time
RUN python -c "from optimum.intel import OVModelForSequenceClassification, OVWeightQuantizationConfig; \
    from transformers import GPT2Tokenizer; \
    model_name = 'debela-arg/dialogpt-am-medium-context'; \
    tokenizer = GPT2Tokenizer.from_pretrained(model_name); \
    qcfg = OVWeightQuantizationConfig(bits=8, ratio=1.0); \
    ov_model = OVModelForSequenceClassification.from_pretrained(model_name, export=True, compile=False, quantization_config=qcfg); \
    ov_model.save_pretrained('/app/model'); \
    tokenizer.save_pretrained('/app/model')"

COPY . .

EXPOSE 5015

CMD ["gunicorn", "--bind", "0.0.0.0:5015", "--workers", "1", "--timeout", "120", "main:app"]
