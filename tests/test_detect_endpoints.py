from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_detect_endpoint_returns_detection_payload(tmp_path: Path):
    image_path = tmp_path / 'sample.jpg'
    image_path.write_bytes(b'fake-image-bytes')

    client = TestClient(app)
    with image_path.open('rb') as handle:
        response = client.post('/detect', files={'file': ('sample.jpg', handle, 'image/jpeg')})

    assert response.status_code == 200
    data = response.json()
    assert data['person'] is True
    assert data['count'] >= 1
    assert data['confidence'] >= 0.0
    assert data['elapsed_ms'] >= 0


def test_detect_file_endpoint_reads_path(tmp_path: Path):
    image_path = tmp_path / 'sample.jpg'
    image_path.write_bytes(b'fake-image-bytes')

    client = TestClient(app)
    response = client.post('/detect-file', json={'path': str(image_path)})

    assert response.status_code == 200
    data = response.json()
    assert data['person'] is True
    assert data['count'] >= 1


def test_detect_file_endpoint_expands_home_directory(tmp_path: Path, monkeypatch):
    home_dir = tmp_path / 'home'
    home_dir.mkdir()
    image_path = home_dir / 'sample.jpg'
    image_path.write_bytes(b'fake-image-bytes')

    monkeypatch.setenv('HOME', str(home_dir))

    client = TestClient(app)
    response = client.post('/detect-file', json={'path': '~/sample.jpg'})

    assert response.status_code == 200
    data = response.json()
    assert data['person'] is True
