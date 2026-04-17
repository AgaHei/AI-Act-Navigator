FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the reranker model at build time
# (avoids cold-start latency on HuggingFace Spaces)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy project source
COPY src/ ./src/
COPY docs/ ./docs/
COPY data/processed/ ./data/processed/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Streamlit configuration for HF Spaces
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
