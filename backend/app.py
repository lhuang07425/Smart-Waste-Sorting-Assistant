"""FastAPI backend for the Smart Waste Sorting Assistant."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .model import Prediction, WasteClassifier

logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Waste Sorting API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: Optional[WasteClassifier] = None

CATEGORY_METADATA: Dict[str, Dict[str, str]] = {
    "recycle": {"emoji": "\u267b\ufe0f", "color": "#0ea5e9"},
    "compost": {"emoji": "\ud83c\udf31", "color": "#22c55e"},
    "trash": {"emoji": "\ud83d\uddd1\ufe0f", "color": "#6b7280"},
}


@app.on_event("startup")
async def load_classifier() -> None:
    global classifier
    try:
        classifier = WasteClassifier()
        logger.info("WasteClassifier loaded successfully")
    except FileNotFoundError as error:
        logger.error("Failed to load classifier: %s", error)
        classifier = None


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)) -> Dict[str, object]:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is unavailable. Train it with backend/train.py.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Unsupported image format") from error

    category, reason, confidence, raw_predictions = classifier.predict(image)
    meta = CATEGORY_METADATA.get(category, CATEGORY_METADATA["trash"])

    predictions_payload: List[Dict[str, object]] = [
        {"label": pred.label, "probability": pred.probability}
        for pred in raw_predictions
    ]

    return {
        "category": category,
        "emoji": meta["emoji"],
        "color": meta["color"],
        "confidence": confidence,
        "reason": reason,
        "predictions": predictions_payload,
    }


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    status = "ready" if classifier is not None else "loading"
    return {"status": status}
