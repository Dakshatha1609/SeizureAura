import requests

def run_model(uploaded_file):
    """Sends EEG CSV file to backend for prediction."""
    files = {'file': uploaded_file}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    return response
