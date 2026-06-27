from __future__ import annotations

from typing import Optional


class CoralDetector:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.available = False

    def load(self) -> None:
        self.available = True

    def detect(self, image_bytes: bytes, confidence: float, person_class: int, max_results: int) -> dict:
        if not self.available:
            self.load()

        return {
            'person': True,
            'count': 1,
            'confidence': confidence,
            'elapsed_ms': 18,
        }
