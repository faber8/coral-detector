import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

CONFIDENCE = float(os.getenv('CONFIDENCE', '0.65'))
PERSON_CLASS = int(os.getenv('PERSON_CLASS', '0'))
MAX_RESULTS = int(os.getenv('MAX_RESULTS', '10'))

MODEL_PATH = MODELS_DIR / 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS_PATH = MODELS_DIR / 'coco_labels.txt'
