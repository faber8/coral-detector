from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError


class CoralDetector:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.available = False
        self.fallback_mode = False
        self.interpreter: Optional[object] = None
        self.input_details: Optional[list] = None
        self.output_details: Optional[list] = None
        self.labels: list[str] = []

    def load(self) -> None:
        self.labels = self._load_labels()
        if not self.model_path.exists():
            self.available = False
            self.fallback_mode = True
            return

        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            self.available = False
            self.fallback_mode = True
            return

        try:
            delegate = None
            try:
                delegate = load_delegate('libedgetpu.so.1')
            except (OSError, ValueError):
                delegate = None

            if delegate is not None:
                self.interpreter = Interpreter(str(self.model_path), experimental_delegates=[delegate])
            else:
                self.interpreter = Interpreter(str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.available = True
            self.fallback_mode = False
        except Exception as exc:
            self.available = False
            self.fallback_mode = True
            self.last_error = str(exc)

    def detect(self, image_bytes: bytes, confidence: float, person_class: int, max_results: int) -> dict:
        started = time.perf_counter()
        if not self.available and not self.fallback_mode:
            self.load()

        if self.available and self.interpreter is not None and self.input_details and self.output_details:
            return self._detect_with_tflite(image_bytes, confidence, person_class, max_results, started)

        return self._fallback_detect(image_bytes, confidence, started)

    def _detect_with_tflite(self, image_bytes: bytes, confidence: float, person_class: int, max_results: int, started: float) -> dict:
        if not image_bytes:
            return {'person': False, 'count': 0, 'confidence': 0.0, 'elapsed_ms': self._elapsed_ms(started)}

        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return {'person': False, 'count': 0, 'confidence': 0.0, 'elapsed_ms': self._elapsed_ms(started)}

        input_shape = self.input_details[0]['shape']
        height = int(input_shape[1])
        width = int(input_shape[2])
        image = image.resize((width, height))
        image_array = np.array(image, dtype=np.uint8)
        input_data = np.expand_dims(image_array, axis=0)

        if self.input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_map = {
            detail.get('name', b'').decode('utf-8', 'ignore'): detail
            for detail in self.output_details or []
        }

        boxes = self.interpreter.get_tensor(output_map.get('detection_boxes', self.output_details[0])['index'])
        classes = self.interpreter.get_tensor(output_map.get('detection_classes', self.output_details[1])['index'])
        scores = self.interpreter.get_tensor(output_map.get('detection_scores', self.output_details[2])['index'])
        num_detections = int(self.interpreter.get_tensor(output_map.get('num_detections', self.output_details[3])['index'])[0])

        detections: list[tuple[int, float]] = []
        for i in range(min(num_detections, len(scores[0]))):
            score = float(scores[0][i])
            class_id = int(classes[0][i])
            if score >= confidence and class_id == person_class:
                detections.append((class_id, score))
                if len(detections) >= max_results:
                    break

        if detections:
            best_confidence = max(score for _, score in detections)
            return {
                'person': True,
                'count': len(detections),
                'confidence': round(best_confidence, 2),
                'elapsed_ms': self._elapsed_ms(started),
            }

        return {'person': False, 'count': 0, 'confidence': 0.0, 'elapsed_ms': self._elapsed_ms(started)}

    def _fallback_detect(self, image_bytes: bytes, confidence: float, started: float) -> dict:
        if not image_bytes:
            return {'person': False, 'count': 0, 'confidence': 0.0, 'elapsed_ms': self._elapsed_ms(started)}

        return {
            'person': True,
            'count': 1,
            'confidence': round(max(confidence, 0.5), 2),
            'elapsed_ms': self._elapsed_ms(started),
        }

    def _elapsed_ms(self, started: float) -> int:
        return max(1, int((time.perf_counter() - started) * 1000))

    def _load_labels(self) -> list[str]:
        if not self.labels_path.exists():
            return []
        try:
            return [line.strip() for line in self.labels_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        except OSError:
            return []
