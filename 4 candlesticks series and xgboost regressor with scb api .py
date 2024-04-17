


import pandas as pd




data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C'], 'day': ['01', '01', '01'], 'month': [4, 3, 2], 'year': [2024, 2024, 2024]}
dfmonthly = pd.DataFrame(data)


yearmonth = dfmonthly['year'].astype(str) + '-' + dfmonthly['month'].astype(str)# + '-' + dfmonthly['day']

print(yearmonth)
dfmonthly = dfmonthly.set_index(yearmonth)

dfmonthly.rename_axis('yearmonth', axis=0, inplace=True)  # inplace=True modifies the DataFrame


print(dfmonthly)




def generate_daily_dates(year, month):
    import datetime
    days_in_month = (datetime.datetime(year, month + 1, 1) - datetime.datetime(year, month, 1)).days

    month_return_dataframe = pd.date_range(start=pd.Timestamp(year=year, month=month, day=1),
                        periods=days_in_month).to_frame(name='dates')

    month_return_dataframe['year'] = year

    month_return_dataframe['month'] = month

    return month_return_dataframe



appended_df = pd.DataFrame()


for i in dfmonthly.index:
    year = int(dfmonthly.loc[i,'year'])
    month = int(dfmonthly.loc[i,'month'])
    daily_dates = generate_daily_dates(year, month)
    merged_df = dfmonthly.merge(daily_dates,  on=['year', 'month'])
    appended_df = pd.concat([appended_df, merged_df], ignore_index=False)


print(appended_df)






















# Function to generate daily dates for a month
def generate_daily_dates(year, month):
    print(pd.to_datetime([year, month, 1]))

    # days_in_month = pd.to_datetime([year, month, 1]).dt.days_in_month.item()
    # return pd.date_range(start=pd.Timestamp(year=year, month=month, day=1), periods=days_in_month)

generate_daily_dates('2024', '04')

# Generate daily dates for the month in monthly DataFrame's date
daily_dates = generate_daily_dates(dfmonthly['month'].dt.month.item(), dfmonthly['year'].dt.year.item())

print(daily_dates)
# Create DataFrame with daily dates
daily_df['date'] = daily_dates

# Outer merge with daily DataFrame as left table (for replication)
merged_df = daily_df.merge(monthly_df, how='outer', left_on='date', right_on='date')

# Fill missing values (NaN) in monthly columns with the single monthly value
merged_df.update(merged_df[['col1', 'col2']].fillna(method='ffill'))

print(merged_df)