import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

CONFIDENCE = float(os.getenv('CONFIDENCE', '0.65'))
PERSON_CLASS = int(os.getenv('PERSON_CLASS', '0'))
MAX_RESULTS = int(os.getenv('MAX_RESULTS', '10'))

MODEL_PATH = MODELS_DIR / 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS_PATH = MODELS_DIR / 'coco_labels.txt'


def ensure_model_files() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        try:
            urllib.request.urlretrieve(
                'https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                MODEL_PATH,
            )
        except Exception:
            pass
    if not LABELS_PATH.exists():
        try:
            LABELS_PATH.write_text('person\n', encoding='utf-8')
        except Exception:
            pass
