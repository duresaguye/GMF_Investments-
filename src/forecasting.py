from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import pickle

def train_arima(ticker):
    df = pd.read_csv(f"data/processed/{ticker}_processed.csv", index_col="Date", parse_dates=True)
    df = df.dropna()

    model = ARIMA(df["Adj Close"], order=(5,1,0))
    model_fit = model.fit()
    with open(f"models/arima_{ticker}.pkl", "wb") as f:
        pickle.dump(model_fit, f)
    return model_fit

if __name__ == "__main__":
    for ticker in ["TSLA"]:
        train_arima(ticker)
