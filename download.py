import requests
import os
import json
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define API endpoints
API_ENDPOINTS = {
    'quiz_submission': 'https://api.jsonserve.com/rJvd7g',
    'quiz': 'https://www.jsonkeeper.com/b/LLQT',  # Updated URL
    'historical_data': 'https://api.jsonserve.com/XgAgFJ'
}

# Create a directory to save the files
os.makedirs("quiz_data", exist_ok=True)

# Download and save data locally
for key, url in API_ENDPOINTS.items():
    try:
        # Bypass SSL verification for endpoints with SSL issues
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()  # Parse JSON data
        
        # Save to file
        file_path = os.path.join("quiz_data", f"{key}.json")
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Data from {key} saved to {file_path}")
    except Exception as e:
        print(f"Failed to download data from {key}: {e}")
