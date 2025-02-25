import requests


def test_frontend():
    response = requests.get("http://frontend:8501/")
    assert response.status_code == 200
