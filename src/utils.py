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
# ---- New utility functions extracted from garch.py ----

def get_portfolio_prices(symbols, start_date, end_date) -> pd.DataFrame:
    """Fetch adjusted prices for portfolio symbols"""
    return fetch_stock_data(symbols, start_date, end_date)


def compute_portfolio_returns(prices: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Compute weighted portfolio log returns and append 'portfolio' column"""
    returns = calculate_returns(prices)
    portfolio = sum(returns[symbol] * weight for symbol, weight in weights.items())
    returns["portfolio"] = portfolio
    return returns

def plot_log_returns(returns_df: pd.DataFrame):
    """Plot log returns for each symbol including portfolio using plotnine"""
    from plotnine import ggplot, aes, geom_line, facet_wrap, labs, theme, element_text

    df = returns_df.reset_index().melt(
        id_vars=[prices.index.name or "index"], var_name="Symbol", value_name="Return")
    p = (
        ggplot(df, aes(x=df.columns[0], y="Return", color="Symbol"))
        + facet_wrap('Symbol', scales='free_y')
        + geom_line()
        + labs(title='Log Returns', x='Date', y='Log Return')
        + theme(
            plot_title=element_text(size=14, face='bold'),
            axis_title=element_text(size=12),
            axis_text=element_text(size=10)
        )
    )
    return p


# %%

def fit_and_summarize_garch(returns_series: pd.Series, p=1, q=1):
    """Fit GARCH model, log summary, and return result object"""
    results = fit_garch(returns_series * 100, p=p, q=q)
    logger.info(results.summary())
    return results


def diagnostics_plots(results: ARCHModelResult, returns_idx):
    """Generate standardized residuals plot and Q-Q plot"""
    std_resid = pd.Series(
        results.resid / np.sqrt(results.conditional_volatility), index=returns_idx
    )
    # Residuals plot
    resid_fig, resid_ax = plt.subplots(figsize=(12, 4))
    resid_ax.plot(returns_idx, std_resid)
    resid_ax.set_title("Standardized Residuals")
    resid_ax.set_xlabel("Date")
    resid_ax.set_ylabel("Residual")
    plt.tight_layout()

    # Q-Q plot
    qq_fig, qq_ax = plt.subplots(figsize=(6, 4))
    stats.probplot(std_resid, dist="norm", plot=qq_ax)
    qq_ax.set_title("Q-Q Plot of Standardized Residuals")
    plt.tight_layout()

    return resid_fig, qq_fig

# %%
def volatility_forecast(results: ARCHModelResult, returns_idx, horizon=5) -> pd.DataFrame:
    """Generate horizon-step volatility forecasts"""
    variance = results.forecast(horizon=horizon).variance.iloc[-1]
    dates = pd.date_range(returns_idx[-1], periods=horizon + 1)[1:]
    return pd.DataFrame({"Date": dates, "Volatility": np.sqrt(variance)})


def plot_volatility_history_and_forecast(results: ARCHModelResult, returns_series: pd.Series, horizon=5):
    """Plot last 100 days of volatility history and future forecasts"""
    hist = pd.Series(np.sqrt(results.conditional_volatility[-100:]), index=returns_series.index[-100:])
    hist_df = pd.DataFrame({"Date": hist.index, "Volatility": hist.values, "Type": "Historical"})
    fc_df = volatility_forecast(results, returns_series.index, horizon)
    fc_df["Type"] = "Forecast"
    combined = pd.concat([hist_df, fc_df])
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, grp in combined.groupby('Type'):
        ax.plot(grp['Date'], grp['Volatility'], label=label)
    ax.set_title("Volatility: History vs Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend()
    plt.tight_layout()
    return fig

# %%
