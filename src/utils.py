import logging

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from plotnine import aes, element_text, geom_line, ggplot, labs, theme
from scipy import stats
from statsmodels.tsa.stattools import acf as acf_stats

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


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, **kwargs):
    """
    Fit GARCH(p,q) model to returns
    """
    model = arch_model(returns, p=p, q=q, **kwargs)
    results = model.fit(disp="off")
    return results


def plot_volatility(results, returns: pd.Series, scale=1000):
    """
    Plot the conditional volatility using plotnine
    """
    # Create a DataFrame for plotting
    volatility_df = pd.DataFrame(
        {"Date": returns.index, "Volatility": np.sqrt(results.conditional_volatility/scale)}
    )

    # Create the plot using plotnine
    volatility_plot = (
        ggplot(volatility_df, aes(x="Date", y="Volatility"))
        + geom_line()
        + labs(title="Conditional Volatility", x="Date", y="Volatility")
        + theme(
            plot_title=element_text(size=14, face="bold"),
            axis_title=element_text(size=12),
            axis_text=element_text(size=10),
        )
    )

    return volatility_plot


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


def calculate_acf_table(series: pd.Series, nlags: int = 15):
    """
    Calculate autocorrelations of squared series, Q-statistics, and p-values.

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
    squared_series = series**2

    # Calculate autocorrelations
    ac = acf_stats(squared_series, nlags=nlags, qstat=True)

    # Create DataFrame
    results = pd.DataFrame(
        {
            "AC": ac[0][1:],  # autocorrelations
            "Q-Stat": ac[1],  # Q-statistics
            "Prob": ac[2],  # p-values
        }
    )

    # Format the index starting from 1
    results.index = range(1, nlags + 1)

    # Round the values
    results = results.round(3)

    return results


def calculate_VaR_returns(returns, conditional_volatility, n_sd):
    mu = np.mean(returns)
    VaR_returns = mu - conditional_volatility*n_sd
    return VaR_returns


def count_exceedances(VaR_returns, returns, perc=False):
    n_exc = sum(returns < VaR_returns)
    if perc:
        perc_exc = n_exc/len(returns) 
        return n_exc, perc_exc
    return n_exc
