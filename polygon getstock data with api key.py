import pandas as pd
import requests

# Define your API key (replace 'YOUR_API_KEY' with your actual key)
API_KEY = 'OPJKUfGJTCIFwBaDX_zEHWbGl6U8Tp2y'

# Set the endpoint URL for historical stock data (e.g., Apple Inc.)
symbol = 'AAPL'
url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2020-01-01/2020-12-31?apiKey={API_KEY}'

# Make the API request
response = requests.get(url)
data = response.json()

print(data)