




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

print(data)


# Extract relevant data (e.g., timestamp, open, close, volume)
df = pd.DataFrame(data['results'])
df = df[['t', 'o', 'c', 'v']]
df.columns = ['timestamp', 'open', 'close', 'volume']

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

## Print the first few rows of the DataFrame
#print(df.head())



import pandas_datareader.data as web
ff = web.DataReader('AAPL',data_source='yahoo')
aapl = web.DataReader(data_source="fred")

get_data_yahoo

print(ff)



from pandas_datareader import data as pdr

all_symbols = pdr.get_nasdaq_symbols()

all_symbols_iex = pdr.get_iex_symbols(API_KEY)

tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
data = web.DataReader(tickers, 'yahoo')
#print(all_symbols)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10) 

import pandas as pd
df = pd.DataFrame(all_symbols)

print(df.T)

unique_values_b = df['Security Name'].unique()
#print(unique_values_b)

"""
unique_values_all_columns = pd.concat([df[col].unique() for col in df.columns])
#print(unique_values_all_columns)
"""

#print(df['Security Name'].unique())

unique_security_names = df['Security Name'].drop_duplicates().tolist()

unique_symbol = df['NASDAQ Symbol'].drop_duplicates().tolist()

df['combined'] = df['NASDAQ Symbol'] + '---' + df['Security Name'].astype(str)


unique_symbol_sec_name = df['combined'].drop_duplicates().tolist()

print(df.keys())

#print(unique_security_names/p)



def print_elements_in_steps(x, p):
    #compartment = 
    for i in range(0, len(x), p):
        print(x[i-p:i])
        input("Press Enter to continue...")


print_elements_in_steps(unique_security_names, 200)

print_elements_in_steps(unique_symbol,500)

print_elements_in_steps(unique_symbol_sec_name,500)
#ctrl c to break

all_security_names =1
# Concatenate the unique security names Fidelity Corporate Bond ETF
for i in  unique_security_names:
    #print(i)
    if i.lower() == 'aapl'.lower() :
        #all_security_names = ', '.join(unique_security_names)

# Print the concatenated list
        #print(all_security_names)
        print(i.lower())