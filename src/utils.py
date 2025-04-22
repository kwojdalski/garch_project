import logging

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from arch import arch_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
