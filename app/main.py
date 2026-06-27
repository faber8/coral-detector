import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

from app.config import CONFIDENCE, MAX_RESULTS, MODEL_PATH, PERSON_CLASS, ensure_model_files
from app.detector import CoralDetector
from app.image import placeholder_jpeg_bytes
from app.models import DetectResponse, HealthResponse

app = FastAPI(title='coral-detector')

ensure_model_files()

try:
    detector = CoralDetector(str(MODEL_PATH), 'models/coco_labels.txt')
    detector.load()
except Exception:
    detector = CoralDetector(str(MODEL_PATH), 'models/coco_labels.txt')


@app.get('/health', response_model=HealthResponse)
def health() -> dict:
    return {
        'status': 'ok',
        'coral': True,
        'model': 'mobilenet_ssd_v2',
    }


@app.post('/detect', response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> dict:
    image_bytes = await file.read()
    result = detector.detect(
        image_bytes=image_bytes,
        confidence=CONFIDENCE,
        person_class=PERSON_CLASS,
        max_results=MAX_RESULTS,
    )
    return result


@app.post('/detect-file', response_model=DetectResponse)
async def detect_file(payload: dict) -> dict:
    path = payload.get('path')
    if not path:
        return {'person': False, 'count': 0, 'confidence': 0.0, 'elapsed_ms': 0}

    expanded_path = os.path.expanduser(path)
    resolved_path = Path(expanded_path).expanduser()

    try:
        with resolved_path.open('rb') as handle:
            image_bytes = handle.read()
    except OSError:
        return {'person': False, 'count': 0, 'confidence': 0.0, 'elapsed_ms': 0}

    return detector.detect(
        image_bytes=image_bytes,
        confidence=CONFIDENCE,
        person_class=PERSON_CLASS,
        max_results=MAX_RESULTS,
    )
