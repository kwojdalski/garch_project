import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from arch import arch_model
from arch.univariate.base import ARCHModelResult


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_stock_data(
    symbols: str | list, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series | pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance

    Parameters:
    -----------
    symbols : str or list
        Single stock symbol or list of stock symbols
    start_date : datetime
        Start date for data retrieval
    end_date : datetime
        End date for data retrieval

    Returns:
    --------
    pandas.Series or pandas.DataFrame
        Close prices for the requested symbol(s)
    """
    if isinstance(symbols, list):
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        return data["Close"]
    else:
        stock = yf.Ticker(symbols)
        df = stock.history(start=start_date, end=end_date)
        return df["Close"]


def calculate_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculate log returns from price series
    """
    return np.log(prices / prices.shift(1)).dropna()


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> ARCHModelResult:
    """
    Fit GARCH(p,q) model to returns
    """
    model = arch_model(returns, p=p, q=q, vol="Garch", dist="normal")
    results = model.fit(disp="off")
    return results


def plot_volatility(
    results: ARCHModelResult, returns: pd.Series
) -> plt.Figure:
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


# Calculate portfolio statistics
def calculate_portfolio_stats(returns_data: pd.DataFrame) -> pd.DataFrame:
    stats_dict = {}
    for column in returns_data.columns:
        series = returns_data[column]
        stats_dict[column] = {
            "Mean": series.mean(),
            "Std. Dev.": series.std(),
            "Skewness": stats.skew(series.dropna()),
            "Kurtosis": stats.kurtosis(series.dropna())
            + 3,  # Adding 3 to get regular kurtosis instead of excess kurtosis
        }

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_dict).round(4)
    return stats_df

######## Extentions #######
# %% [markdown]

# %%
