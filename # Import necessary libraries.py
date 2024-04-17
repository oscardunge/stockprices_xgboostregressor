# Import necessary libraries
import sqlite
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score

# Download MSFT stock data
#msft = yf.download("MSFT", start="2020-01-01", end="2024-01-01")

# Create features (e.g., moving averages, volatility)
# Preprocess data (handle missing values, etc.)














"""
# Define features and target variable
X = msft[["Open", "High", "Low", "Volume"]]  # Use relevant columns
y = msft["Close"]  # Closing price as the target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate precision (minimize false positives)
precision = precision_score(y_test > y_pred, y_test < y_pred)

print(f"Model Precision: {precision:.2f}")

# Now you can use the trained model to predict future stock prices!
"""