import requests

def test_server_is_up():
    """
    Test to verify that the FastAPI server is running and responding with HTTP 200.
    """
    response = requests.get("http://app:8000/health")
    assert response.status_code == 200
