FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# System dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# -------------------------------------------------
# Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# Pre-download GOT-OCR2_0 model
# -------------------------------------------------
ENV HF_HOME=/models/hf

RUN python - <<'EOF'
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

print("GOT-OCR2_0 downloaded successfully")
EOF

# -------------------------------------------------
# App
# -------------------------------------------------
COPY handler.py .

CMD ["python", "-u", "handler.py"]
