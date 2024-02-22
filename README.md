Prediction
'''
import requests
import pandas as pd

# Load your CSV file
data = pd.read_csv("RawData/train.csv")

# URL of your Flask app's predict endpoint
url = "https://obesitypredictgitaction-production.up.railway.app/predict" # Sample deployment URL

# Convert DataFrame to CSV string
csv_data = data.to_csv(index=False)

# Send POST request with the CSV data
files = {'file': ('data.csv', csv_data)}
response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    predictions = response.json()['predictions']
    print("Predictions:", predictions)
else:
    print("Error:", response.json())
'''

