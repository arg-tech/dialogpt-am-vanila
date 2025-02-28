# Use an official Python runtime as a parent image
FROM python:3.8.2-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies separately to ensure compatibility
RUN pip install --no-cache-dir transformers torch scikit-learn

# Create and set working directory
WORKDIR /app
COPY requirements.txt .

# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Preload the Hugging Face model and save it to /app/model
RUN python -c "from transformers import AutoModelForSequenceClassification, GPT2Tokenizer; \
    model_name = 'debela-arg/dialogtp-am-medium'; \
    model = AutoModelForSequenceClassification.from_pretrained(model_name); \
    tokenizer = GPT2Tokenizer.from_pretrained(model_name); \
    model.save_pretrained('/app/model'); \
    tokenizer.save_pretrained('/app/model')"

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 5015

# Set the default command for the container
CMD ["python", "./main.py"]
