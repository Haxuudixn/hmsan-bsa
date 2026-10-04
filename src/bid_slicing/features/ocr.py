"""EasyOCR wrapper for extracting text from document image blocks.

Design:
  - Lazy singleton so the heavy OCR engine is loaded only once per process.
  - Accepts PIL Image or numpy array; returns joined recognized text.
  - Falls back to PaddleOCR if EasyOCR is unavailable (but EasyOCR models
    are already cached in this environment, so it is the default).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

_engine: Any | None = None
_engine_kind: str = ""


def _to_image_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _get_engine(use_gpu: bool = False, prefer_paddleocr: bool = False) -> Any:
    """Return the shared OCR engine.

    By default EasyOCR is preferred when available. Pass prefer_paddleocr=True
    to force PaddleOCR (needed for inference parity with training data).
    """
    global _engine, _engine_kind
    if _engine is not None:
        if not prefer_paddleocr or _engine_kind == "paddleocr":
            return _engine
        _engine = None
        _engine_kind = ""

    if prefer_paddleocr:
        from paddleocr import PaddleOCR
        _engine = PaddleOCR(
            lang="ch",
            engine="onnxruntime",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _engine_kind = "paddleocr"
        return _engine

    # Try EasyOCR first: its ch_sim model is already cached locally.
    try:
        import easyocr
        _engine = easyocr.Reader(
            ["ch_sim", "en"],
            gpu=use_gpu,
            verbose=False,
        )
        _engine_kind = "easyocr"
        return _engine
    except Exception as exc:  # pragma: no cover - fallback path
        import warnings
        warnings.warn(f"EasyOCR unavailable ({exc}); falling back to PaddleOCR")

    from paddleocr import PaddleOCR
    _engine = PaddleOCR(
        lang="ch",
        engine="onnxruntime",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    _engine_kind = "paddleocr"
    return _engine


def extract_text(
    image: Image.Image | np.ndarray,
    use_gpu: bool = False,
    prefer_paddleocr: bool = False,
) -> str:
    """Extract text from a single image and join lines with newline."""
    engine = _get_engine(use_gpu, prefer_paddleocr=prefer_paddleocr)
    arr = _to_image_array(image)

    if _engine_kind == "easyocr":
        lines = engine.readtext(arr, detail=0, paragraph=False)
        return "\n".join(str(t) for t in lines if str(t).strip())

    results = engine.predict(arr)
    lines: list[str] = []
    for res in results:
        texts = _extract_texts_from_paddle_result(res)
        lines.extend(t for t in texts if t and t.strip())
    return "\n".join(lines)


def extract_batch(
    images: list[Image.Image | np.ndarray],
    use_gpu: bool = False,
    prefer_paddleocr: bool = False,
) -> list[str]:
    """Extract text from multiple images sequentially (safer for CPU)."""
    engine = _get_engine(use_gpu, prefer_paddleocr=prefer_paddleocr)
    outputs: list[str] = []

    for image in images:
        arr = _to_image_array(image)
        if _engine_kind == "easyocr":
            lines = engine.readtext(arr, detail=0, paragraph=False)
            outputs.append("\n".join(str(t) for t in lines if str(t).strip()))
        else:
            results = engine.predict(arr)
            lines = []
            for res in results:
                lines.extend(t for t in _extract_texts_from_paddle_result(res)
                             if t and t.strip())
            outputs.append("\n".join(lines))
    return outputs


def _extract_texts_from_paddle_result(res: Any) -> list[str]:
    """Best-effort extraction from a PaddleOCR v3 result object."""
    if hasattr(res, "rec_texts"):
        return [str(t) for t in res.rec_texts]
    if isinstance(res, dict) and "rec_texts" in res:
        return [str(t) for t in res["rec_texts"]]
    try:
        payload = res.json if callable(getattr(res, "json", None)) else res.json
        if isinstance(payload, dict) and "res" in payload:
            inner = payload["res"]
            if isinstance(inner, dict) and "rec_texts" in inner:
                return [str(t) for t in inner["rec_texts"]]
    except Exception:
        pass
    return []