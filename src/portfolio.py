import pandas as pd
import numpy as np
from scipy.optimize import minimize

def optimize_portfolio():
    tickers = ["TSLA", "BND", "SPY"]
    data = {t: pd.read_csv(f"data/processed/{t}_processed.csv", index_col="Date", parse_dates=True)["Returns"] for t in tickers}
    df = pd.DataFrame(data).dropna()

    returns = df.mean() * 252  # Annualized returns
    cov_matrix = df.cov() * 252  # Annualized covariance

    def sharpe_ratio(weights):
        port_return = np.dot(weights, returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_volatility  # Minimize negative Sharpe Ratio

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = [(0, 1) for _ in range(len(tickers))]
    initial_weights = [1/len(tickers)] * len(tickers)

    optimal_weights = minimize(sharpe_ratio, initial_weights, bounds=bounds, constraints=constraints).x
    return dict(zip(tickers, optimal_weights))

if __name__ == "__main__":
    print("Optimal Portfolio Weights:", optimize_portfolio())
