import pandas as pd

import pandas_datareader as pdr

tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
# tickers = ['AAPL']


import plotly.graph_objects as pl

# Assuming 'df' is your DataFrame containing the financial data
fig = pl.Figure() 

# Fetch data for each symbol
for ticker in tickers:
    df = pdr.get_data_stooq(ticker)
    df['Ticker'] = ticker  # Add a new column with the ticker name
    # df = pd.concat([df, data])
    print(f'({df.head(5)})')
    # Print the first 5 rows of data for each symbol
    fig.add_trace(pl.Candlestick(
        x=df.index,  # Use the index (date) as the x-axis
        open=df['Open'],  # Open price for candlestick body
        high=df['High'],  # High price for upper wick
        low=df['Low'],   # Low price for lower wick
        close=df['Close'],  # Close price for candlestick body color
        name=ticker  # Assign the name to this time series
    ))
    
    


fig.update_layout(
    title='Candlestick Chart for Financial Data',
    xaxis_title='Date',
    yaxis_title='Price',
)








# First XGBoost model for Pima Indians dataset
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
# import numpy as np

# dataset = np.loadtxt('C:\Users\oscar\Downloads\pima-indians-diabetes.csv', delimiter=",", encoding='utf-8')





import numpy as np

# dataset = np.loadtxt(r'C:\Users\oscar\Downloads\pima-indians-diabetes.csv', delimiter=",")


# print(dataset)

# print(dataset[0:,0:8])

# split data into X and y
X = df.index.values
Y = df['Close'].values

print(X)
print(Y)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data


print(X_train)
print(y_train)


model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))








fig.show()