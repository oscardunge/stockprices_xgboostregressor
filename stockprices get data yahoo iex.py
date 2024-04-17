




import pandas as pd
import requests

# Define your API key (replace 'YOUR_API_KEY' with your actual key)
API_KEY = 'wo9qgDuM2iaeFYFPWzl32OBvc92LKDjrncEUGUm5'

# Set the endpoint URL for historical stock data (e.g., Apple Inc.)
symbol = 'AAPL'
url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/month/2020-01-01/2020-12-31?apiKey={API_KEY}'

# Make the API request
response = requests.get(url)
data = response.json()

#print(data)


# Extract relevant data (e.g., timestamp, open, close, volume)
df = pd.DataFrame(data['results'])
df = df[['t', 'o', 'c', 'v']]
df.columns = ['timestamp', 'open', 'close', 'volume']

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

## Print the first few rows of the DataFrame
#print(df.head())



import pandas_datareader.data as web
ff = web.datareader("aapl","yahoo")
aapl = web.datareader("aapl", "iex")

#print(aapl)



from pandas_datareader import data as pdr

all_symbols = pdr.get_nasdaq_symbols()

#print(all_symbols)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10) 

import pandas as pd
df = pd.DataFrame(all_symbols)
unique_values_b = df['Security Name'].unique()
#print(unique_values_b)

"""
unique_values_all_columns = pd.concat([df[col].unique() for col in df.columns])
#print(unique_values_all_columns)
"""

#print(df['Security Name'].unique())

unique_security_names = df['Security Name'].drop_duplicates().tolist()

print(unique_security_names)

all_security_names =1
# Concatenate the unique security names Fidelity Corporate Bond ETF
for i in  unique_security_names:
    #print(i)
    if i.lower() == 'aapl'.lower() :
        #all_security_names = ', '.join(unique_security_names)

# Print the concatenated list
        #print(all_security_names)
        print(i.lower())