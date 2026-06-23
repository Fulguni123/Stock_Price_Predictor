import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

print("Downloading stock data...")

data = yf.download("AAPL", start="2020-01-01", end="2025-01-01", auto_adjust=True)

# Close prices
close_prices = pd.DataFrame()
close_prices["Close"] = data["Close"].squeeze()

# Previous day's close
close_prices["Previous_Close"] = close_prices["Close"].shift(1)
close_prices.dropna(inplace=True)

X = close_prices[["Previous_Close"]]
y = close_prices["Close"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Error
mse = mean_squared_error(y_test, predictions)

# Latest closing price
latest_price = float(close_prices["Close"].iloc[-1])

# Predict next day
next_price = model.predict(
    pd.DataFrame({"Previous_Close": [latest_price]})
)

print("\n===== STOCK PRICE PREDICTOR =====")
print("Mean Squared Error:", round(mse, 2))
print("Latest Closing Price:", round(latest_price, 2))
print("Predicted Next Day Price:", round(float(next_price[0]), 2))