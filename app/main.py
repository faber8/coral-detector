import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response

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


@app.get('/', response_class=HTMLResponse)
def web_form() -> str:
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Coral Detector</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 2rem;}
            input[type=file] {margin-bottom: 1rem;}
            .result {margin-top: 1rem; padding: 1rem; border: 1px solid #ddd; background: #f9f9f9;}
        </style>
    </head>
    <body>
        <h1>Coral Detector</h1>
        <form id="uploadForm">
            <label for="file">Sélectionnez une image :</label><br>
            <input type="file" id="file" name="file" accept="image/*" required/><br>
            <button type="submit">Analyser</button>
        </form>
        <div class="result" id="result"></div>

        <script>
            const form = document.getElementById('uploadForm');
            const resultEl = document.getElementById('result');

            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                resultEl.textContent = 'Analyse en cours...';
                const fileInput = document.getElementById('file');
                const file = fileInput.files[0];
                if (!file) {
                    resultEl.textContent = 'Aucun fichier sélectionné.';
                    return;
                }
                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        body: formData,
                    });
                    if (!response.ok) {
                        throw new Error('Erreur réseau ' + response.status);
                    }
                    const data = await response.json();
                    resultEl.innerHTML = `<strong>Résultat :</strong><br>
                        Personne : ${data.person}<br>
                        Count : ${data.count}<br>
                        Confidence : ${data.confidence}<br>
                        Elapsed ms : ${data.elapsed_ms}`;
                } catch (error) {
                    resultEl.textContent = 'Erreur : ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    '''


@app.get('/health', response_model=HealthResponse)
def health() -> dict:
    return {
        'status': 'ok',
        'coral': True,
        'model': 'mobilenet_ssd_v2',
    }


@app.get('/debug')
def debug() -> dict:
    return {
        'model_exists': str(MODEL_PATH.exists()),
        'model_path': str(MODEL_PATH),
        'labels_exists': str(MODEL_PATH.exists()),
        'available': detector.available,
        'fallback_mode': detector.fallback_mode,
        'last_error': getattr(detector, 'last_error', None),
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
