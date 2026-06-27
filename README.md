# coral-detector

Application FastAPI de détection de personnes avec un modèle Coral-compatible.

## Démarrage local

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

- GET /health
- POST /detect

## Docker

```bash
docker compose up --build
```

L’API sera disponible sur http://localhost:8000/docs
