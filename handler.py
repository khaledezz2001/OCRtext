import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_PATH = "/models/hf/models--reducto--RolmOCR"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def load_model():
    global processor, model
    if processor is not None and model is not None:
        return

    log("Loading RolmOCR processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    log("Loading RolmOCR model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model.eval()
    log("RolmOCR model loaded")


def decode_image(b64: str) -> Image.Image:
    image_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def handler(event):
    log("Handler called")
    load_model()

    image_b64 = event["input"]["image"]
    image = decode_image(image_b64)

    log("Running OCR")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048)

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    log("OCR finished")

    return {
        "text": text,
        "format": "markdown"
    }


runpod.serverless.start({
    "handler": handler
})
#
