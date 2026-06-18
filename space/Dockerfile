FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --chown=user requirements.txt .

# Prebuilt CPU wheel for llama-cpp-python (no local compile needed)
RUN pip install --no-cache-dir --user \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
