import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start="2015-01-01", end="2025-01-31"):
    stock = yf.download(ticker, start=start, end=end)
    stock.to_csv(f"data/raw/{ticker}.csv")
    return stock

if __name__ == "__main__":
    for ticker in ["TSLA", "BND", "SPY"]:
        fetch_stock_data(ticker)
