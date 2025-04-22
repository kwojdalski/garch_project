import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy import stats
from statsmodels.tsa.stattools import acf as acf_stats
from tabulate import tabulate

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


def calculate_returns(prices: pd.Series | pd.DataFrame):
    """
    Calculate log returns from price series
    """
    return np.log(prices / prices.shift(1)).dropna()


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    """
    Fit GARCH(p,q) model to returns
    """
    model = arch_model(returns, p=p, q=q, vol="Garch", dist="normal")
    results = model.fit(disp="off")
    return results


def plot_volatility(results, returns: pd.Series):
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
def calculate_portfolio_stats(returns_data: pd.DataFrame):
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


def calculate_acf_table(returns: pd.Series, nlags: int = 15):
    """
    Calculate autocorrelations of squared portfolio returns, Q-statistics, and p-values.

    Parameters
    ----------
    returns : pd.Series
        Time series of portfolio returns
    nlags : int, optional
        Number of lags to calculate, by default 15

    Returns
    -------
    pd.DataFrame
        Table with autocorrelations, Q-statistics, and p-values
    """
    # Calculate squared returns
    squared_returns = returns**2

    # Calculate autocorrelations
    ac = acf_stats(squared_returns, nlags=nlags, qstat=True)

    # Create DataFrame
    results = pd.DataFrame(
        {
            "AC": ac[0],  # autocorrelations
            "Q-Stat": ac[1],  # Q-statistics
            "Prob": ac[2],  # p-values
        }
    )

    # Format the index starting from 1
    results.index = range(1, nlags + 1)

    # Round the values
    results = results.round(3)

    return results


def display_acf_table(returns: pd.Series, nlags: int = 15):
    """
    Display the autocorrelation table in a formatted way.

    Parameters
    ----------
    returns : pd.Series
        Time series of portfolio returns
    nlags : int, optional
        Number of lags to calculate, by default 15
    """

    table = calculate_acf_table(returns, nlags)

    # Add lag column to the table
    display_table = table.copy()
    display_table.insert(0, "Lag", range(1, nlags + 1))

    print("Autocorrelations of Squared Portfolio Returns")
    print(
        tabulate(
            display_table,
            headers=["Lag", "AC", "Q-Stat", "Prob"],
            floatfmt=(".0f", ".3f", ".2f", ".3f"),
            tablefmt="simple",
        )
    )
    print(
        f"Sample: {returns.index[0].strftime('%B %d, %Y')} to {returns.index[-1].strftime('%B %d, %Y')}."
    )
