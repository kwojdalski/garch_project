# ---
# title: GARCH Model Implementation
# author: Based on Engle (2001)
# format:
#   html:
#     toc: true
#     toc-depth: 3
#     code-fold: true
#     theme: cosmo
#     highlight-style: github
# execute:
#   echo: true
#   warning: false
# ---

# %%
# | label: setup
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_line,
    ggplot,
    labs,
    scale_color_manual,
    theme,
    theme_minimal,
)
from scipy import stats

if os.getcwd().endswith("src"):
    os.chdir("..")

from src.utils import (
    calculate_acf_table,
    calculate_portfolio_stats,
    calculate_returns,
    fetch_stock_data,
    fit_garch,
    plot_volatility,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% [markdown]
# # Data Preparation
#
# ## Portfolio Composition
#
# This portfolio consists of:
#
# - 50% Nasdaq (^IXIC)
# - 30% Dow Jones Industrial Average (^DJI)
# - 20% 10-year Treasury Constant Maturity Rate [^1]
#
# [^1]: Potentially ^TNX, this needs to be checked further on.
#
# ## Fetching Stock Data
#
# We'll fetch the portfolio components data using our implementation function:

# %%
# | label: fetch-data
# Parameters
# Nasdaq, Dow Jones, and 10-year Treasury
symbols = ["^IXIC", "^DJI", "^TNX"]  #
# Define the date range, based on the paper
# Sample period
start_date = datetime(1990, 3, 22)
end_date = datetime(2000, 3, 24)

# Portfolio weights, taken from the article
weights = {
    "^IXIC": 0.50,  # Nasdaq
    "^DJI": 0.30,  # Dow Jones
    "^TNX": 0.20,  # 10-year Treasury
}


# Fetch data using our implementation
logger.info("Fetching data...")
prices = fetch_stock_data(symbols, start_date, end_date)

# Display the first few rows
prices.tail()

# Apparently, DJIA data is missing for 1990-1992, so we'll drop it
prices = prices.drop(columns=["^DJI", "^TNX"])

# %% [markdown]
# ### TNX and DJIA
#
# As the data was missing for 1990-1992, we had to take it from somewhere else.
# We found Dow Jones Industrial Average (^DJI) data [here](https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data).
#
# For RATE we used a good proxy which we found here looks like a good proxy for TNX, we'll data we found[here](https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DGS10&scale=left&cosd=2020-04-17&coed=2025-04-17&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=liin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-04-22&revision_date=2025-04-22&nd=1962-01-02)
#
# ####  DJIA data retrieval

# %%
djia_prices = pd.read_csv(
    "data/dja-performance-report-daily.csv",
    index_col="Effective Date",
    parse_dates=True,
)
djia_prices = djia_prices.rename(columns={"Close Value": "^DJI"})
prices = prices.join(djia_prices["^DJI"])

# %% [markdown]
# ####  RATE data retrieval
#
# Data has been obtained from the hyperlink

# %%
tnote_yield = pd.read_csv(
    "data/DGS10.csv",
    index_col="observation_date",
    parse_dates=True,
)
tnote_yield = tnote_yield.rename(columns={"DGS10": "^TNX"})

# tnote yield is not exactly the price, but we merge it anyway
prices = prices.join(tnote_yield["^TNX"])

# %% [markdown]
# ## Calculating Returns
#
# Next, we calculate the log returns using our implementation function:

# %%
# | label: calculate-returns
returns = calculate_returns(prices)

portfolio_prices = pd.Series(0, index=prices.index, dtype=float)

# Apply weights to each component
for symbol, weight in weights.items():
    # Use pandas multiplication and addition
    portfolio_returns = portfolio_prices.add(returns[symbol].multiply(weight))

# Add portfolio returns to the returns DataFrame
returns["portfolio"] = portfolio_returns

# %% [markdown]
# ## Visualizing Price and Returns
#
# Let's visualize returns for all data series and compare with the plot from the article

# %%
# | label: visualize-data
# | fig-cap: Nasdaq, Dow Jones and Bond Returns
# | fig-subcap:
# |   - Nasdaq
# |   - Dow Jones
# |   - Bond
returns_df = pd.DataFrame(returns).reset_index()
returns_df = pd.melt(
    returns_df, id_vars=["Date"], var_name="Symbol", value_name="Return"
)
returns_plot = (
    ggplot(returns_df, aes(x="Date", y="Return", color="Symbol"))
    + facet_wrap("Symbol", scales="free_y")
    + geom_line(color="#FF9900")
    + labs(title="Log Returns", x="Date", y="Log Return")
    + theme(
        plot_title=element_text(size=14, face="bold"),
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
    )
)


returns_plot

# %% [markdown]
# ## Portfolio Statistics

# %%
# | label: tbl-portfolio-stats
# | tbl-cap: Portfolio Statistics
# | tbl-cap-location: top
portfolio_stats = calculate_portfolio_stats(returns)
portfolio_stats

# %% [markdown]
# The table above shows the summary statistics for our portfolio components and the overall portfolio.
# The statistics include mean returns, standard deviation, skewness, and kurtosis for each component.
#
# Our results show minor differences compared to Table 1 in Engle (2001):
#
# * The statistics for Nasdaq (^IXIC) and Dow Jones (^DJI) differ by only 1-2 basis points
# * Larger discrepancies appear for the Rate component, likely because the article lacks specificity about the exact data source used
#     + The most notable differences are in skewness (0.38 vs -0.20) and kurtosis (4.96 vs 5.96)
# * These differences in the Rate component are the primary reason for the slight variation between our replicated portfolio and the one presented in the paper
#
# ## ACF of Residuals

# %%
# | label: acf-residuals
# | fig-cap: ACF of Residuals
# | fig-subcap:
# |   - Portfolio Returns
# |   - Portfolio Residuals
acf_plot = calculate_acf_table(returns["portfolio"])
acf_plot

# %% [markdown]
# # GARCH Model Estimation
#
# ## Fitting the GARCH(1,1) Model
#
# Now we'll fit a GARCH(1,1) model to the returns data using our implementation:

# %%
# | label: fit-garch
# Fit GARCH(1,1) model using our implementation
logger.info("Fitting GARCH(1,1) model...")
results = fit_garch(returns["portfolio"])

# Display model summary
logger.info("\nModel Summary:")
logger.info(results.summary())


# %% [markdown]
# ## Model Parameters
#
# The GARCH(1,1) model is specified as:
#
# $$\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$
#
# where:
# - $\sigma_t^2$ is the conditional variance
# - $\omega$ is the constant term
# - $\alpha_1$ is the ARCH effect
# - $\beta_1$ is the GARCH effect
# - $\varepsilon_{t-1}^2$ is the squared lagged returns
# - $\sigma_{t-1}^2$ is the lagged conditional variance
#
# Let's extract and display the model parameters:

# %%
# | label: model-parameters
# Extract parameters
params = results.params
print("Model Parameters:")
for param, value in params.items():
    logger.info(f"{param}: {value:.6f}")


# %%
# | label: tbl-squared-residuals
# | tbl-cap: Autocorrelations of Squared Standardized Residuals
# | tbl-cap-location: top
acf_plot = calculate_acf_table(results.resid / np.sqrt(results.conditional_volatility))
acf_plot

# %% [markdown]
# # Volatility Analysis
#
# ## Conditional Volatility
#
# Let's plot the conditional volatility using our implementation:

# %%
# | label: plot-volatility
# | fig-cap: Conditional Volatility

# Create volatility plot using our implementation
volatility_plot = plot_volatility(results, returns)

# %% [markdown]
# ## Model Diagnostics
#
# Let's examine the model residuals using plotnine:

# %%
# | label: model-diagnostics
# | fig-cap: Standardized Residuals and Q-Q Plot

# Get standardized residuals
std_resid = results.resid / np.sqrt(results.conditional_volatility)

# Create dataframe for residuals
resid_df = pd.DataFrame({"Date": std_resid.index, "Residual": std_resid.values})

# Create residuals plot
resid_plot = (
    ggplot(resid_df, aes(x="Date", y="Residual"))
    + geom_line(color="#FF9900")
    + labs(title="Standardized Residuals", x="Date", y="Standardized Residual")
    + theme_minimal()
    + theme(
        plot_title=element_text(size=14, face="bold"),
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
    )
)

# For Q-Q plot, we need to use matplotlib as plotnine doesn't have a direct Q-Q plot
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot residuals using plotnine
print(resid_plot)

# Q-Q plot using matplotlib
stats.probplot(std_resid, dist="norm", plot=ax2)
ax2.set_title("Q-Q Plot of Standardized Residuals")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Volatility Forecasting
#
# ## Generating Forecasts
#
# Let's generate volatility forecasts:

# %%
# | label: forecast
# Generate forecasts
logger.info("Generating volatility forecast...")
forecast = results.forecast(horizon=5)
logger.info("\nVolatility Forecast:")
logger.info(forecast.variance.iloc[-1])

# %% [markdown]
# ## Visualizing Forecasts
#
# Let's visualize the forecast using plotnine:

# %%
# | label: plot-forecast
# | fig-cap: Volatility Forecast

# Create dataframes for historical and forecast data
historical_df = pd.DataFrame(
    {
        "Date": returns.index[-100:],
        "Volatility": np.sqrt(results.conditional_volatility[-100:]),
        "Type": "Historical",
    }
)

forecast_dates = pd.date_range(returns.index[-1], periods=6)[1:]
forecast_df = pd.DataFrame(
    {
        "Date": forecast_dates,
        "Volatility": np.sqrt(forecast.variance.iloc[-1]),
        "Type": "Forecast",
    }
)

# Combine dataframes
combined_df = pd.concat([historical_df, forecast_df])

# Create forecast plot
forecast_plot = (
    ggplot(combined_df, aes(x="Date", y="Volatility", color="Type"))
    + geom_line()
    + labs(title="Volatility Forecast", x="Date", y="Volatility")
    + scale_color_manual(values=["#3366CC", "#FF9900"])
    + theme_minimal()
    + theme(
        plot_title=element_text(size=14, face="bold"),
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
        legend_title=element_blank(),
    )
)

# Display plot
forecast_plot
