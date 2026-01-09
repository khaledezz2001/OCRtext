import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===============================
# OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def decode_image(b64: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    image.thumbnail((2048, 2048), Image.BICUBIC)
    return image


def load_model():
    global processor, model
    if model is not None:
        return

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    log("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=True
    )

    model.eval()
    log("RolmOCR model loaded successfully")


def handler(event):
    log("Handler called")
    load_model()

    image = decode_image(event["input"]["image"])

    # ✅ CORRECT QWEN2.5-VL MESSAGE FORMAT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Read all text in this image."}
            ]
        }
    ]

    # ✅ THIS INJECTS IMAGE TOKENS (CRITICAL)
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512
        )

    result = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return {
        "text": result.strip(),
        "format": "markdown"
    }


log("Preloading model...")
load_model()

runpod.serverless.start({"handler": handler})
