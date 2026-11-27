from __future__ import annotations

import numpy as np
from app.models.clip_encoder import OpenCLIPEncoder

ATTRIBUTE_PROMPTS: dict[str, dict[str, str]] = {
    "shape": {
        "round": "a jewelry ring featuring a round brilliant cut diamond",
        "oval": "a jewelry ring featuring an oval cut diamond",
        "emerald": "a jewelry ring featuring a rectangular emerald step cut diamond",
        "cushion": "a jewelry ring featuring a cushion cut diamond",
        "pear": "a jewelry ring featuring a teardrop pear cut diamond",
        "marquise": "a jewelry ring featuring an elongated marquise cut diamond",
        "princess": "a jewelry ring featuring a square princess cut diamond",
        "radiant": "a jewelry ring featuring a radiant cut diamond",
        "asscher": "a jewelry ring featuring a square asscher cut diamond",
    },
    "metal": {
        "yellow-gold": "yellow gold diamond jewelry",
        "white-gold": "white gold or platinum diamond jewelry",
        "rose-gold": "rose gold diamond jewelry",
    },
    "category": {
        "engagement-rings": "a diamond engagement ring",
        "wedding-rings": "a wedding band ring",
        "necklaces": "a diamond necklace pendant",
        "earrings": "a pair of diamond earrings",
        "bracelets": "a diamond bracelet",
    },
}


class ZeroShotAttributeScorer:
    """Computes zero-shot attribute predictions from an image embedding using OpenCLIP text prototypes."""

    def __init__(self, clip_encoder: OpenCLIPEncoder) -> None:
        self.prototypes: dict[str, tuple[list[str], np.ndarray]] = {}
        for group, label_map in ATTRIBUTE_PROMPTS.items():
            labels = list(label_map.keys())
            prompts = list(label_map.values())
            text_embs = clip_encoder.encode_texts(prompts)
            self.prototypes[group] = (labels, text_embs)

    def predict(self, image_embedding: np.ndarray) -> dict[str, dict[str, float | str]]:
        predictions: dict[str, dict[str, float | str]] = {}
        for group, (labels, text_embs) in self.prototypes.items():
            logits = text_embs @ image_embedding
            # Softmax with CLIP temperature scale
            scaled = (logits - np.max(logits)) * 20.0
            probs = np.exp(scaled)
            probs = probs / np.sum(probs)
            best_idx = int(np.argmax(probs))
            predictions[group] = {
                "value": labels[best_idx],
                "confidence": round(float(probs[best_idx]), 4),
            }
        return predictions
