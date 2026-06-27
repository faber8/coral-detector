from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_ok():
    client = TestClient(app)
    response = client.get('/health')

    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert data['coral'] is True
    assert data['model'] == 'mobilenet_ssd_v2'
