"""Waste classification model fine-tuned on TrashNet."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import models, transforms


MODELS_DIR = Path(__file__).resolve().parent / "models"
WEIGHTS_FILE = MODELS_DIR / "trashnet_resnet18.pth"
META_FILE = MODELS_DIR / "trashnet_meta.json"


@dataclass
class Prediction:
    label: str
    probability: float


class WasteClassifier:
    """Load the fine-tuned TrashNet model and map results to waste categories."""

    def __init__(self, device: str | None = None) -> None:
        if not WEIGHTS_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError(
                "Fine-tuned weights or metadata not found. Run backend/train.py first to train the model."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        meta = json.loads(META_FILE.read_text())
        self.class_names: List[str] = meta["class_names"]
        self.category_map: Dict[str, str] = {k.lower(): v for k, v in meta["category_map"].items()}
        self.fallback_category: str = meta.get("fallback_category", "recycle")
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self) -> torch.nn.Module:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(in_features, len(self.class_names)),
        )
        state_dict = torch.load(WEIGHTS_FILE, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Tuple[str, str, float, List[Prediction]]:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        prob_values, indices = torch.topk(probabilities, k=min(3, probabilities.size(0)))
        predictions = [
            Prediction(label=self.class_names[idx], probability=float(prob_values[i]))
            for i, idx in enumerate(indices.tolist())
        ]
        category, reason, confidence = self._map_predictions(predictions)
        return category, reason, confidence, predictions

    def _map_predictions(self, preds: List[Prediction]) -> Tuple[str, str, float]:
        best = preds[0]
        label = best.label.lower()
        category = self.category_map.get(label)
        if category is None:
            category = "trash" if label == "trash" else self.fallback_category
        reason = f"Model predicted '{best.label}'"
        return category, reason, best.probability


__all__ = ["WasteClassifier", "Prediction"]
