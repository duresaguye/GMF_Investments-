from fetch_data import fetch_stock_data
from preprocess import preprocess_data
from forecasting import train_arima
from portfolio import optimize_portfolio

def main():
    print("Fetching data...")
    for ticker in ["TSLA", "BND", "SPY"]:
        fetch_stock_data(ticker)

    print("Preprocessing data...")
    for ticker in ["TSLA", "BND", "SPY"]:
        preprocess_data(ticker)

    print("Training forecasting model for TSLA...")
    train_arima("TSLA")

    print("Optimizing portfolio...")
    optimal_weights = optimize_portfolio()
    print("Optimal Portfolio Weights:", optimal_weights)

if __name__ == "__main__":
    main()
