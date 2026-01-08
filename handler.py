import base64
import io
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model ONCE (cold start)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()


def decode_image(b64: str) -> Image.Image:
    image_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def handler(event):
    """
    Expected input:
    {
      "input": {
        "image": "<base64 image>"
      }
    }
    """
    try:
        image_b64 = event["input"]["image"]
        image = decode_image(image_b64)

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096
            )

        text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        return {
            "text": clean_text(text),
            "format": "markdown"
        }

    except Exception as e:
        return {"error": str(e)}
