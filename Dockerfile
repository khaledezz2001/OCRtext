# 🔴 IMPORTANT: force x86_64 for RunPod GPU
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-runtime-ubuntu22.04

# -----------------------------
# Basic setup
# -----------------------------
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------
# ✅ FIX: Hugging Face cache location (prevents "No space left on device")
# -----------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# -----------------------------
# Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# App code
# -----------------------------
COPY handler.py .

# -----------------------------
# RunPod serverless entry
# -----------------------------
CMD ["python", "-u", "handler.py"]
