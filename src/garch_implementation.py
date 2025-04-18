from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from arch import arch_model


def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    """
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df["Close"]


def calculate_returns(prices):
    """
    Calculate log returns from price series
    """
    return np.log(prices / prices.shift(1)).dropna()


def fit_garch(returns, p=1, q=1):
    """
    Fit GARCH(p,q) model to returns
    """
    model = arch_model(returns, p=p, q=q, vol="Garch", dist="normal")
    results = model.fit(disp="off")
    return results


def plot_volatility(results, returns):
    """
    Plot the conditional volatility
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(returns.index, np.sqrt(results.conditional_volatility))
    ax.set_title("Conditional Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    plt.tight_layout()
    return fig


def main():
    # Parameters
    symbol = "^GSPC"  # S&P 500 index
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data

    # Fetch data
    print("Fetching data...")
    prices = fetch_stock_data(symbol, start_date, end_date)
    returns = calculate_returns(prices)

    # Fit GARCH(1,1) model
    print("\nFitting GARCH(1,1) model...")
    results = fit_garch(returns)
    print("\nModel Summary:")
    print(results.summary())

    # Plot volatility
    print("\nGenerating volatility plot...")
    fig = plot_volatility(results, returns)
    plt.savefig("volatility_plot.png")
    plt.close()

    # Forecast
    print("\nGenerating volatility forecast...")
    forecast = results.forecast(horizon=5)
    print("\nVolatility Forecast:")
    print(forecast.variance.iloc[-1])


if __name__ == "__main__":
    main()
