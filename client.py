import requests
import json

url = 'http://127.0.0.1:5000/predict'

data = {
    'input_folder': './test_images',
    'output_folder': './pred_results'
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Prediction completed successfully.")
else:
    print(f"Error: {response.json().get('error')}")
