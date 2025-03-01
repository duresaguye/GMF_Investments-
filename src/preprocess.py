import pandas as pd

def preprocess_data(ticker):
    df = pd.read_csv(f"data/raw/{ticker}.csv", index_col="Date", parse_dates=True)
    df = df.dropna()  # Remove missing values
    df["Returns"] = df["Adj Close"].pct_change()  # Compute daily returns
    df.to_csv(f"data/processed/{ticker}_processed.csv")
    return df

if __name__ == "__main__":
    for ticker in ["TSLA", "BND", "SPY"]:
        preprocess_data(ticker)
