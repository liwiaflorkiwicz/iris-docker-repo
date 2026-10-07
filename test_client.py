import requests

# API endpoint
url = "http://localhost:8000/predict"

# Payload matching the expected features
payload = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

# Send the POST request to the Flask API
response = requests.post(url, json=payload)

# Print the parsed JSON response
print("Status code:", response.status_code)
print("Response JSON:", response.json())