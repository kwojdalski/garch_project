# %% [markdown]
# ---
# title: GARCH Model Implementation
# author:
#   - name: "Krzysztof Wojdalski"
#   - name: "Piotr"
#   - name: "Shah"
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
# jupyter: python3
# ---
#
#
# # Introduction
#
# ## Background
#
# This project aims to replicate the GARCH model analysis presented in Robert Engle's 2001 Nobel Prize lecture, "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics." The lecture demonstrates how GARCH models can be used to analyze and forecast financial market volatility.
#
# ### Rationale behind the Study
#
# 1) From the initial inspection of the article, easy to obtain data
# 2) Well-known author so we asssumed that the methodology is sound and well-explained
# 3) The author's research has had significant influence and impact in the field of time series analysis
# 4) Research could be easily extended for different datasets / portfolios
#
#
# ### Steps in the Research
# The replication focuses on the following:
#
# 1. Constructing a portfolio similar (or, hopefully, identical) to the one used in Engle's paper
# 2. Calculating and analyzing portfolio returns
# 3. Fitting a GARCH(1,1) model to the portfolio returns
# 4. Examining the model's performance in capturing volatility clustering
# 5. Generating volatility forecasts in the same manner as Engle did
#
# By following Engle's methodology, the project provides a practical implementation of GARCH modeling techniques for financial time series analysis.
#
#
# # Imports and Setup

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

if os.getcwd().endswith("src") or os.getcwd().endswith("notebooks"):
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
# #### Problems with the data
#
# Engle described the data set as follows:
#
# > Let's use the GARCH(1,1) tools to estimate the 1 percent
#  value at risk of a $1,000,000 portfolio on March 23, 2000. This portfolio consists of
#  50 percent Nasdaq, 30 percent DowJones and 20 percent long bonds. The long
#  bond is a ten-year constant maturity Treasury. The portfolio has constant proportions of wealth
#  in each asset that would entail some rebalancing over time.
#
# Due to vagueness of the claim above, we'll try to reverse engineer the portfolio, but it's not
# a perfect solution. When we calculate return on the portfolio, we implictly assume daily rebalancing, i.e.,
# weights are constant.
#
# ## Fetching Stock Data
#
# We'll fetch the portfolio components data using the implemented function:
#
# - The function is implemented in `src/utils.py`
# - it uses `yfinance` - a Python package for downloading stock data from Yahoo Finance
# - start and end dates are taken from the article
# - weights for portfolio components are taken from the article, i.e. 50% Nasdaq, 30% Dow Jones, and 20% 10-year Treasury.

# %%
# | label: fetch-data
# Parameters
# Nasdaq, Dow Jones, and 10-year Treasury
symbols = ["^IXIC", "^DJI", "^TNX"]  #
# Define the date range, based on the paper
# Sample period
start_date = datetime(1990, 3, 22)
end_date = datetime(2001, 6, 1)

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
# ### RATE and Dow Jones Industrial Average (^DJI)
#
# As the data for `^DJI` was missing for 1990-1992, we had to take it from somewhere else.
# We found Dow Jones Industrial Average (^DJI) data [here](https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data).
#
# For RATE we used a good proxy which we found here looks like a good proxy for TNX, we'll use data we found [here](https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DGS10&scale=left&cosd=2020-04-17&coed=2025-04-17&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=liin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-04-22&revision_date=2025-04-22&nd=1962-01-02)
#
# ####  DJIA data retrieval

# %%
djia_prices = pd.read_csv(
    "./data/dja-performance-report-daily.csv",
    index_col="dt",
    parse_dates=True,
)
djia_prices = djia_prices.rename(columns={"Close Value": "^DJI"})
prices = prices.join(djia_prices["djia"])


# %% [markdown]
# ####  RATE data retrieval
#
# Data has been obtained from the hyperlink

# %%
tnote_yield = pd.read_csv(
    "./data/DGS10.csv",
    index_col="observation_date",
    parse_dates=True,
)
tnote_yield = tnote_yield.rename(columns={"DGS10": "^TNX"})

# tnote yield is not exactly the price, but we merge it anyway
prices = prices.join(tnote_yield["^TNX"])

# %%
prices.rename({'djia': '^DJI'}, axis=1, inplace=True)
prices

# %% [markdown]
# ## Calculating Returns
#
# This function uses log returns, which are calculated as follows:
#
# $$r_t = \ln \left( \frac{P_t}{P_{t-1}} \right)$$
#
# where $P_t$ is the price at time $t$.

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

# %%
# split the data
returns = returns.loc[:'2000-03-23']
returns_oos = returns.loc['2000-03-24':]

prices = prices.loc[:'2000-03-23']
prices_oos = prices.loc['2000-03-24':]

# %% [markdown]
# ## Visualizing Price and Returns
#
# Let's visualize prices (excl. portfolio) and returns for all data series and compare with the plot from the article.
#
# Visual inspection of the plot below shows that the data is very similar to the one in the article. The only difference is `^TNX` which doesn't overlap perfectly with plots from the article (could be easily spotted by looking at the outliers.)
#
# Possible explanation:
#  The 10-year Treasury Constant Maturity Rate (^TNX) is a model-derived value rather than an actual market price. Since new 10-year Treasury bonds aren't issued daily, the constant maturity yield represents a theoretical value of what a 10-year Treasury security would yield if issued at current market conditions. This value is calculated through interpolation of yields from Treasury securities with similar credit quality but different maturities. While this approximation closely tracks what an actual new 10-year bond would yield, small discrepancies can exist. Nevertheless, the constant maturity rate provides valuable insights into market expectations regarding inflation, economic growth, and future interest rates.
#
# Hence, it's likely that Engle's methodology and / or selection of the securities with the same level of credit riskiness is different than the one used by Federal Reserve Bank of St. Louis in 2025.

# %%
# | label: fig-prices
# | fig-cap: Nasdaq, Dow Jones and Bond Prices
# | fig-subcap:
# |   - 'Sample: March 23, 1990 to March 23, 2000.'

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(prices['^DJI'], color='black')
axes[0].grid(alpha=0.3)
axes[0].set_title('^DJI')

axes[1].plot(prices['^IXIC'], color='black')
axes[1].grid(alpha=0.3)
axes[1].set_title('^IXIC')

axes[2].plot(prices['^TNX'], color='black')
axes[2].grid(alpha=0.3)
axes[2].set_title('^TNX')

plt.show()


# %% [markdown]
# #### Returns from the original paper (for reference)
# ![Returns from the original paper](../data/screenshots/returns.png)

# %%
# | label: fig-returns
# | fig-cap: Nasdaq, Dow Jones and Bond Returns
# | fig-subcap:
# |   - 'Sample: March 23, 1990 to March 23, 2000.'
fig, axes = plt.subplots(2, 2, figsize = (16, 8))

axes[0, 0].plot(returns['^DJI'], color = "black")
axes[0, 0].set_title('^DJI')
axes[0, 0].grid(linestyle = '--', alpha = 0.3)

axes[0, 1].plot(returns['^TNX'], color = "black")
axes[0, 1].set_title('^TNX')
axes[0, 1].grid(linestyle = '--', alpha = 0.3)

axes[1, 0].plot(returns['^IXIC'], color = "black")
axes[1, 0].set_title('^IXIC')
axes[1, 0].grid(linestyle = '--', alpha = 0.3)

axes[1, 1].plot(returns['portfolio'], color = "black")
axes[1, 1].set_title('portfolio')
axes[1, 1].grid(linestyle = '--', alpha = 0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Portfolio Statistics
#
# #### Article Portfolio Statistics (for reference)
# ![Portfolio Statistics](../data/screenshots/portfolio_statistics.png)

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
# ## ACF of Squared Portfolio Returns
#
# #### Article ACF of Squared Portfolio Returns (for reference)
# ![ACF of Squared Portfolio Returns](../data/screenshots/squared_returns.png)

# %%
# | label: tbl-portfolio-returns
# | tbl-cap: Autocorrelations of Squared Portfolio Returns
acf_plot = calculate_acf_table(returns["portfolio"])
acf_plot

# %% [markdown]
# # GARCH Model
#
# ## Model Parameters
#
# The GARCH(1,1) model is specified as:
#
# $$\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$
#
# where:
#
# - $\sigma_t^2$ is the conditional variance
# - $\omega$ is the constant term
# - $\alpha_1$ is the ARCH effect
# - $\beta_1$ is the GARCH effect
# - $\varepsilon_{t-1}^2$ is the squared lagged returns
# - $\sigma_{t-1}^2$ is the lagged conditional variance
#
# ## GARCH Model Estimation
#
# ### Fitting the GARCH(1,1) Model
#
# Now we'll fit a GARCH(1,1) model to the returns
#
# #### Article GARCH(1,1) Model (for reference)
# ![GARCH(1,1) Model](../data/screenshots/garch_one_one.png)

# %%
# | label: tbl-garch-one-one
# | tbl-cap: GARCH(1,1)
# | tbl-cap-location: top

logger.info("Fitting GARCH(1,1) model...")
results = fit_garch(returns["portfolio"])

logger.info(results.summary())
# Extract coefficients, standard errors, z-statistics, and p-values
coef = results.params
std_err = results.std_err
z_stat = coef / std_err
p_values = results.pvalues

# Create a DataFrame to display the results
model_results = pd.DataFrame(
    {
        "Coef": coef,
        "St. Err": std_err,
        "Z-Stat": z_stat,
        "P-Value": p_values,
    }
)
model_results

# %% [markdown]
# ## ACF of Squared Standardized Residuals
#
#
# #### Article ACF of Squared Standardized Residuals (for reference)
# ![ACF of Squared Standardized Residuals](../data/screenshots/squared_standardized_residuals.png)

# %%
# | label: tbl-squared-residuals
# | tbl-cap: Autocorrelations of Squared Standardized Residuals
acf_plot = calculate_acf_table(results.resid / np.sqrt(results.conditional_volatility))
acf_plot

# %% [markdown]
# # Volatility Analysis
#
# ## Conditional Volatility
#
# Let's plot the conditional volatility using our implementation:

# %%
# | label: fig-conditional-volatility
# | fig-cap: Conditional Volatility

volatility_plot = plot_volatility(results, returns)
volatility_plot

# %% [markdown]
# ## Model Diagnostics
#
# Let's examine the model residuals using plotnine:

# %%
# | label: model-diagnostics
# | fig-cap: Standardized Residuals

# Get standardized residuals
std_resid = results.resid / np.sqrt(results.conditional_volatility)

plt.figure(figsize=(12, 6))
plt.plot(std_resid, color='black', lw=1)
plt.grid(alpha=0.3)
plt.title("Standardized residuals")
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
# | label: fig-forecast
# | fig-cap: Volatility Forecast

plt.figure(figsize=(12, 6))
plt.plot(returns['portfolio'][-100:], color='black', lw=1, label='returns')
plt.plot(results.conditional_volatility[-100:], color='red', lw=1, label='GARCH volatility')
plt.grid(alpha=0.3)
plt.legend()
plt.show()


# %% [markdown]
# # References
#
# * Engle, R. F. (2001). GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics. Journal of Economic Perspectives, 15(4), 157-168.
# * Kaggle. (2023). Dow Jones Industrial Average (^DJI) Data. [Dataset]. Available at: https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data
# * St. Louis Fed. (2023). 10-Year Treasury Constant Maturity Rate [^TNX]. [Dataset]. Available at: https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DGS10&scale=left&cosd=2020-04-17&coed=2025-04-17&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=liin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-04-22&revision_date=2025-04-22&nd=1962-01-02
# * Yahoo Finance. (2023). [Dataset]. Available at: https://finance.yahoo.com/quote/%5EIXIC/history?period1=631152000&period2=1713878400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
# * GenAI used for proof checking and type hinting
