
import requests
import pandas as pd

query = {
  "query": [
    {
      "code": "VaruTjanstegrupp",
      "selection": {
        "filter": "vs:VaruTjänstegrCoicopA",
        "values": [
          "01",
          "02",
          "03",
          "04",
          "05",
          "06",
          "07",
          "08",
          "09",
          "10",
          "11",
          "12"
        ]
      }
    },
    {
      "code": "ContentsCode",
      "selection": {
        "filter": "item",
        "values": [
          "000003TJ"
        ]
      }
    },
    {
      "code": "Tid",
      "selection": {
        "filter": "item",
        "values": [
          "2019M05",
          "2019M06",
          "2019M07",
          "2019M08",
          "2019M09",
          "2019M10",
          "2019M11",
          "2019M12",
          "2020M01",
          "2020M02",
          "2020M03",
          "2020M04",
          "2020M05",
          "2020M06",
          "2020M07",
          "2020M08",
          "2020M09",
          "2020M10",
          "2020M11",
          "2020M12",
          "2021M01",
          "2021M02",
          "2021M03",
          "2021M04",
          "2021M05",
          "2021M06",
          "2021M07",
          "2021M08",
          "2021M09",
          "2021M10",
          "2021M11",
          "2021M12",
          "2022M01",
          "2022M02",
          "2022M03",
          "2022M04",
          "2022M05",
          "2022M06",
          "2022M07",
          "2022M08",
          "2022M09",
          "2022M10",
          "2022M11",
          "2022M12",
          "2023M01",
          "2023M02",
          "2023M03",
          "2023M04",
          "2023M05",
          "2023M06",
          "2023M07",
          "2023M08",
          "2023M09",
          "2023M10",
          "2023M11",
          "2023M12",
          "2024M01",
          "2024M02",
          "2024M03"
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
}

# API endpoint URL
url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/PR/PR0101/PR0101A/KPICOI80MN"

# Make the API request
response = requests.post(url, json=query)



data = response.json()["data"]
print(data)

existing_df = pd.DataFrame(data)

value1, value2 = df.index[1].split('_')
print(existing_df)


print(str(existing_df.loc[1,'key']).split('M'))

for i in str(existing_df.loc[1,'key']).split('M'):
    # print(type(i))
    # i.split("']")
    
    yearmonth = i[-4:].split("']")
    # month = i[-4:].split("']")[1]
    # print(type(yearmonth))
    print(yearmonth[0] +'-'+ yearmonth[1])


import pandas as pd

# Sample DataFrame
data = {'col1': ['apple_1234', 'banana_5678', 'cherry_9012']}
df = pd.DataFrame(data)

# Extract last 4 characters
df['last_4'] = df['col1'].str[-4:]

print(df)







del(df)
del(dfappend)
del(fig)
del(X)
del(Y)
del(X_train)
del(X_test)
del(y_pred)
del(y_test)
del(y_train)
del(model)

import os
# os.system("pip install pandasql")
# os.system("pip install pandas_datareader")
# os.system("pip install pandas")
# os.system("pip install plotly")



import pandas as pd

import pandas_datareader as pdr

tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
#tickers = ['AAPL']


import plotly.graph_objects as pl

# Assuming 'df' is your DataFrame containing the financial data
fig = pl.Figure() 

df = pd.DataFrame()

dfappend = pd.DataFrame()

# df.dtypes

i=0
# Fetch data for each symbol
for ticker in tickers:
    i=i+1
    #df = pdr.get_data_stooq(ticker)
    dfappend = pdr.get_data_stooq(ticker)
    # dfappend['Ticker'] = i
    dfappend['Ticker'] = ticker
    # Add a new column with the ticker name
    # df = df.append(dfappend)
    df = pd.concat([df, dfappend], ignore_index=False)
    # df = pd.concat([df, data])
    # print(f'({df.head(5)})')
    # Print the first 5 rows of data for each symbol
    merged_df = existing_df.merge(df, how='outer', on=dfappend.index)
    print(merged_df)
    fig.add_trace(pl.Candlestick(
        x=dfappend.index,  # Use the index (date) as the x-axis
        open=dfappend['Open'],  # Open price for candlestick body
        high=dfappend['High'],  # High price for upper wick
        low=dfappend['Low'],   # Low price for lower wick
        close=dfappend['Close'],  # Close price for candlestick body color
        name=ticker  # Assign the name to this time series
    ))


fig.update_layout(
    title='Candlestick Chart for Financial Data',
    xaxis_title='Date',
    yaxis_title='Price',
)

#merged_df = existing_df.merge(df, how='outer', on=dfappend.index)

#print(df)

# import sqlite3
# print(sqlite3)
# #!pip install sqlite3

# import sqlite3 
# conn = sqlite3.connect('example.db') 
# from pandasql import sqldf 

#fig.show()


# sqldf("select * from df",env=None)

# result_df = sqldf("SELECT * FROM df", globals())

# result_df = sqldf("SELECT * FROM df", locals(), locals())  # Pass df and globals explicitly


# print(result_df)


# for index, row in df.iterrows():
    # print(row['name'], row['age'])





for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")

# print(df)



# First XGBoost model for Pima Indians dataset
# from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
# import numpy as np

# dataset = np.loadtxt('C:\Users\oscar\Downloads\pima-indians-diabetes.csv', delimiter=",", encoding='utf-8')




# fit a final xgboost model on the housing dataset and make a prediction
from numpy import asarray
import numpy as np

# dataset = np.loadtxt(r'C:\Users\oscar\Downloads\pima-indians-diabetes.csv', delimiter=",")

df['Ticker'] = df['Ticker'].astype('category')

import pandas as pd

# Assuming your DataFrame `df` has a categorical column 'category_col'
df = pd.get_dummies(df, columns=['Ticker'])  # One-hot encode the categorical column
df = df.drop('Ticker', axis=1)  # Separate features (X) and 'Ticker' variable (y)
# y = df['Ticker']


# print(dataset)

# print(dataset[0:,0:8])

# split data into X and y
# X = df.index.values
X = df.loc[:, df.columns != 'Close']
Y = df['Close'].values
# Define the data types for each field


# dtype = [('name', 'U255'), ('date', 'datetime64[ns]')]
# for index in df.itertuples():
#     print(df.index.values)
#     # print(name)


# Create the structured array using the DataFrame index and a column
# array = np.array([(name, index) for name, index in df.itertuples()], dtype=dtype)

# X = np.array([(name, index) for name, index in df.itertuples()], dtype=dtype)

# X = asarray([X])
# Y = asarray([Y])


print(len(Y))
type(X)
print(X)
print(Y)


df.dtypes




num_rows = df.shape[0]

print(num_rows)
# split data into train and test sets
seed = 12
test_size = 0.33
type(test_size)


train_size = 1 - test_size
type(train_size)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size, random_state=seed)
# fit model no training data

X_train.shape
X_test
y_train
y_test


# print(X_train)
# print(y_train)



# model = XGBClassifier()
model = XGBRegressor()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)



round_y_test = [round(value) for value in y_test]
round_y_pred = [round(value) for value in y_pred]
print(round_y_pred)
print(round_y_test)

accuracy = accuracy_score(round_y_test, round_y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(accuracy)




