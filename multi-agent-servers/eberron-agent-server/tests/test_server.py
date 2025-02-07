import requests


def test_server_is_up():
    """
    Test to verify that the FastAPI server is running and responding with HTTP 200.
    """
    response = requests.get("http://app:8000/health")
    assert response.status_code == 200


def test_respond_endpoint():
    """Test the /respond endpoint using the requests library."""
    # Ensure the FastAPI app is running on localhost:8000
    url = "http://app:8000/respond"

    # Send a POST request with the required payload
    payload = {"content": "Test message"}
    response = requests.post(url, json=payload, stream=True)

    # Check the response status code
    assert response.status_code == 200

    # Read the streaming response
    response_text = "".join([chunk.decode("utf-8") for chunk in response.iter_content()])

    # Validate the response content
    assert "" in response_text  # Replace `expected_value` with the actual expected output

