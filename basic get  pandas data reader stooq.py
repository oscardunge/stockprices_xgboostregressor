

# import pandas_datareader as pdr
# from datetime import datetime

# # Define the stock symbol and date range
# symbol = 'AAPL'
# start_date = datetime(2010, 1, 1)
# end_date = datetime(2020, 1, 1)

# # Fetch the data
# apple_stock = pdr.get_data_yahoo(symbols=symbol, start=start_date, end=end_date)

# # Print the adjusted closing prices
# print(apple_stock['Adj Close'])

import pandas_datareader as pdr
import yfinance as yf

# Override pandas_datareader with yfinance
#yf.pdr_override()

# Define your list of symbols
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# Fetch data for each symbol
for ticker in tickers:
    data = pdr.get_data_stooq(ticker)
    print(f'({ticker}, :::\n {data.head(5)})')
    # Print the first 5 rows of data for each symbol