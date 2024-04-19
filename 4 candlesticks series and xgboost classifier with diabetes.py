
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

print(existing_df)





# ----------------------------




# value1, value2 = existing_df.index[1].split('_')
for i in existing_df.index:
    yearmonthlist = str(existing_df.loc[i,'key']).split('M')
    year = yearmonthlist[0][-4:]
    month = yearmonthlist[1][-4:-2]
    existing_df.loc[i,'year'] = year
    existing_df.loc[i,'month'] = month
    scbkpisubgroup = str(existing_df.loc[i,'key']).split("['")
    
    scbkpisubgroup = scbkpisubgroup[1][0:2]
    # print(scbkpisubgroup)
    
    existing_df.loc[i,'scbkpisubgroup'] = scbkpisubgroup

# print(existing_df['year'])
def generate_daily_dates_df(year, month):
    import datetime
    days_in_month = (datetime.datetime(year, month + 1, 1) - datetime.datetime(year, month, 1)).days

    month_return_dataframe = pd.date_range(start=pd.Timestamp(year=year, month=month, day=1),
                        periods=days_in_month).to_frame(name='dates')

    month_return_dataframe['year'] = year

    month_return_dataframe['month'] = month

    return month_return_dataframe


print(existing_df)
>>> print(existing_df)
#                key    values  year month scbkpisubgroup
# 0    [01, 2019M05]  [318.87]  2019    05             01
# 1    [01, 2019M06]  [319.21]  2019    06             01
# 2    [01, 2019M07]  [325.89]  2019    07             01
# 3    [01, 2019M08]  [323.01]  2019    08             01
# 4    [01, 2019M09]  [322.68]  2019    09             01
# ..             ...       ...   ...   ...            ...
# 703  [12, 2023M11]  [483.01]  2023    11             12
# 704  [12, 2023M12]  [483.67]  2023    12             12
# 705  [12, 2024M01]  [491.20]  2024    01             12
# 706  [12, 2024M02]  [494.24]  2024    02             12
# 707  [12, 2024M03]  [495.96]  2024    03             12
print(type(existing_df))
print(existing_df[existing_df['month'] == '06'])
print(existing_df.dtypes)


existing_df['kpivalue'] = existing_df['values'] 

column_names = existing_df.columns.tolist()
print(column_names)

for i in range(2,13,1):
    print(i)

import pandas as pd

# Sample DataFrame (replace with your actual data)
# data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'col2': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'A'], 'category_col': ['cat1', 'cat1', 'cat2', 'cat1', 'cat2', 'cat3', 'cat3', 'cat1', 'cat2']}
# df = pd.DataFrame(data)


existing_df['values'] = existing_df['values'].apply(lambda x: x[0])
existing_df['scbkpisubgroup'] = existing_df['scbkpisubgroup'].apply(lambda x: int(x))
existing_df2 = existing_df.set_index(['year', 'month'])



# existing_df = existing_df.set_index(['year', 'month', 'scbkpisubgroup'])
# existing_df = existing_df.set_index(['year', 'month', 'scbkpisubgroup'])

# existing_df.set_index(new_index, inplace=True)

print(existing_df)

print(existing_df2)

pivot_var = 'scbkpisubgroup'
value_cols = 'values'

existing_df2.drop('key', axis=1, inplace=True)


pivoted_df = existing_df2.set_index([pivot_var, existing_df2.index]).unstack(fill_value=None)  # Replace -11414 with a missing value indicator (optional)
print(pivoted_df.T)
# pivoted_dataframe = existing_df2.pivot_table(index=pivot_var, values=value_cols)

# del transposed_dataframe

# transposed_dataframe = existing_df[existing_df['scbkpisubgroup'] == 1 ]

# transposed_dataframe2 = existing_df.set_index(['year', 'month', 'scbkpisubgroup'])



# print(transposed_dataframe)

# for i in existing_df.index.get_level_values('scbkpisubgroup'):

#     transposed_dataframe
#     transposed_dataframe = pd.concat([transposed_dataframe, existing_df], ignore_index=False)
#     print(existing_df.index[i])
#     # transposed_dataframe = existing_df[existing_df[i] == 1]






# print(transposed_dataframe)

# Split by category (assuming 12 unique values)
for index, row in existing_df.iterrows():
  value = row['scbkpisubgroup']  # Extract the value from the column
  print(value)
  for i in range(2,13,1):
      if value == i:
          transposed_dataframe = pd.concat([transposed_dataframe, existing_df], ignore_index=False)
          # transposed_dataframe = transposed_dataframe.merge(existing_df,  on=['year', 'month'])

print(transposed_dataframe)

# Access individual DataFrames by category
for category, df_part in split_dfs.items():
  print(f"DataFrame for category: {category}")
  print(df_part)

for i in existing_df['key']:
    for i in range(1,12,1):
        if existing_df.loc[i,'kpisubgroup'] == 
    
    
    
import numpy as np

# def convert_to_float(x):
#   try:
#     # Remove leading/trailing spaces
#     return float(x[0].strip().lstrip('0'))  # Explicitly remove leading zeros
#   except (ValueError, IndexError):
#     return np.nan  # Return NaN for errors

# existing_df['values'] = existing_df['values'].apply(convert_to_float)


# try:
#   existing_df['values'] = pd.to_numeric(existing_df['values'], errors='coerce')  # Coerce errors to NaN
# except:
#   pass

def get_first_value_as_string(x):
  return str(x[0])  # Explicitly cast to string

# Reshape with custom aggregation
out = df.pivot_table(values='values', index=['year', 'month'], columns='scbkpisubgroup', aggfunc=get_first_value_as_string)

# Now 'out' contains the entire string values
print(out)

out = existing_df.pivot_table(values='kpivalue', index=['year', 'month'], columns='scbkpisubgroup', fill_value=0)
print(out)
print(type(out))


for i in 
appended_df = pd.concat([appended_df, merged_df], ignore_index=False)


daily_dates_df = generate_daily_dates_df(2024, 3)
print(daily_dates_df)
print(existing_df)
merged_df = existing_df.merge(daily_dates_df,  on=['year', 'month'])

for i in existing_df.index:
    print(existing_df.loc[i])
    year = int(existing_df.loc[i,'year'])
    month = int(existing_df.loc[i,'month'])
    print(year)
    print(month)
    # daily_dates_df = generate_daily_dates_df(year, month)
    # print(daily_dates_df)
    # merged_df = existing_df.merge(daily_dates_df,  on=['year', 'month'])
    # print(merged_df)
    # appended_df = pd.concat([appended_df, merged_df], ignore_index=False)


# print(appended_df)



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
    print(f'({df.head(5)})')
    # Print the first 5 rows of data for each symbol
    merged_df = existing_df.merge(df, how='outer', on=['year', 'month'])
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




