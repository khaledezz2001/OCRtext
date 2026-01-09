import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===============================
# ENV FIXES (VERY IMPORTANT)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # disable parallel temp downloads

# ===============================
# Configuration
# ===============================
MODEL_ID = "reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def decode_image(b64: str) -> Image.Image:
    image_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ===============================
# Load model ONCE
# ===============================
def load_model():
    global processor, model

    if model is not None:
        return

    log("Loading RolmOCR processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True
    )

    log("Loading RolmOCR model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    model.eval()
    log("RolmOCR model loaded successfully")


# ===============================
# RunPod Handler
# ===============================
def handler(event):
    log("Handler called")
    load_model()

    if "image" not in event["input"]:
        return {"error": "Missing image in input"}

    image = decode_image(event["input"]["image"])

    log("Running OCR")

    prompt = "Read all text in this image."

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048
        )

    text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    log("OCR finished")

    return {
        "text": text.strip(),
        "format": "markdown"
    }


# ===============================
# Start worker
# ===============================
runpod.serverless.start({
    "handler": handler
})
