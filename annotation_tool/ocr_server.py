"""Local OCR server for the annotation tool.

Run from the project root:
    python annotation_tool/ocr_server.py

Then the annotation tool can call POST /ocr with an image file (multipart
field `image`) and receive `{"text": "..."}`.

CORS is enabled so the file:// or localhost:5500 annotation page can call it.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from bid_slicing.features.ocr import extract_text

app = FastAPI(title="Bid Doc OCR Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_endpoint(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        return {"text": "", "error": "empty image"}

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        text = extract_text(img)
        return {"text": text}
    except Exception as exc:
        return {"text": "", "error": str(exc)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8502)