# %% [markdown]
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
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .qmd
#       format_name: quarto
#       format_version: '1.0'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /usr/local/share/jupyter/kernels/python3
# ---

# %%
# | label: setup for vscode/mac
import logging
from datetime import datetime
import os, sys

#for mac
root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

print("Import paths:", sys.path[:3])
#####
#%%
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

from src.utils import (
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
# - 50% Nasdaq (^IXIC)
# - 30% Dow Jones Industrial Average (^DJI)
# - 20% 10-year Treasury Constant Maturity Rate (^TNX)
#
# ## Fetching Stock Data
#
# We'll fetch the portfolio components data using our implementation function:

# %%
# | label: fetch-data
# Parameters
symbols = ["^IXIC", "^DJI", "^TNX"]  # Nasdaq, Dow Jones, and 10-year Treasury
# Define the date range from Robert Engle's paper (2001)
# Sample period from Table 2 and Table 3: March 23, 1990 to March 23, 2000
start_date = datetime(2004, 3, 22)
end_date = datetime(2024, 3, 24)


# start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
# end_date = datetime.now()


# Portfolio weights
weights = {
    "^IXIC": 0.50,  # Nasdaq
    "^DJI": 0.30,  # Dow Jones
    "^TNX": 0.20,  # 10-year Treasury
}


# Fetch data using our implementation
logger.info("Fetching data...")
prices = fetch_stock_data(symbols, start_date, end_date)

# Display the first few rows
prices.dtypes

file_path = "/Users/shah/garch_project/src/prices.pkl"
prices.to_pickle(file_path)
# Apparently, DJIA data is missing for 1990-1992, so we'll drop it
## prices = prices.drop(columns=["^DJI"])
## prices = prices.drop(columns=["^TNX"])
# ... we have to take it from somewhere else
# found it here: https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data
# For RATE which looks we'll use
# https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DGS10&scale=left&cosd=2020-04-17&coed=2025-04-17&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=liin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-04-22&revision_date=2025-04-22&nd=1962-01-02
# %%
prices.columns

#%%
# %%
"""
djia_prices = pd.read_csv(
    "data/dja-performance-report-daily.csv",
    index_col="Effective Date",
    parse_dates=True,
)
djia_prices = djia_prices.rename(columns={"Close Value": "^DJI"})
prices = prices.join(djia_prices["^DJI"])

"""
# %%
djia_prices = pd.read_pickle("/Users/shah/garch_project/src/prices.pkl")[['^DJI']]
djia_prices.index = pd.to_datetime(djia_prices.index)
print(djia_prices.head(5))

prices = pd.read_pickle("/Users/shah/garch_project/src/prices.pkl")
print(prices.head(5))

# %%
"""
tnote_yield = pd.read_csv(
    "data/DGS10.csv",
    index_col="observation_date",
    parse_dates=True,
)
tnote_yield = tnote_yield.rename(columns={"DGS10": "^TNX"})

# tnote yield is not exactly the price, but we merge it anyway
prices = prices.join(tnote_yield["^TNX"])

"""
# %% [markdown]
# ## Calculating Returns
#
# Next, we calculate the log returns using our implementation function:

# %%
# | label: calculate-returns
# Calculate log returns using our implementation
returns = calculate_returns(prices)

portfolio_prices = pd.Series(0, index=prices.index, dtype=float)

# Apply weights to each component
for symbol, weight in weights.items():
    # Use pandas multiplication and addition
    portfolio_returns = portfolio_prices.add(returns[symbol].multiply(weight))
portfolio_returns.head()
# Add portfolio returns to the returns DataFrame
returns["portfolio"] = portfolio_returns
returns["portfolio"]
returns.head()



# Table 1: Portfolio Data
portfolio_stats = calculate_portfolio_stats(returns)

# Display the table with proper formatting
print("\nTable 1")
print("Portfolio Data")
print("=" * 80)
print(portfolio_stats.to_string())
print("\nSample: March 23, 1990 to March 23, 2000")

# %% [markdown]
# The table above shows the summary statistics for our portfolio components and the overall portfolio.
# The statistics include mean returns, standard deviation, skewness, and kurtosis for each component.
#
# It differs marginally from Table 1 in Engle (2001):,
# * by 1-2 bps for each stat for Nasdaq (^IXIC) and Dow Jones (^DJI)
# * slightly more for Rate (the article is inadequately precise in terms what exactly has been used)
#     + especially for skewness and kurtosis (0.38 vs -0.20, and 4.96 vs 5.96, respectively)
# * Rate is the root cause of slight difference between Portfolio and our replicated portfolio

# %% [markdown]
# ## Visualizing Price and Returns
#
# Let's visualize both the price series and returns using plotnine:

# %%
# | label: visualize-data
# | fig-cap: S&P 500 Price and Log Returns

# Create dataframes for plotting


# Create dataframe for returns plotting
returns_df = pd.DataFrame(returns).reset_index()
returns_df = pd.melt(
    returns_df, id_vars=["Date"], var_name="Symbol", value_name="Return"
)
# %%
# Create returns plot
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
# # GARCH Model Estimation
#
# ## Fitting the GARCH(1,1) Model
#
# Now we'll fit a GARCH(1,1) model to the returns data using our implementation:

# %%
# | label: fit-garch
# Fit GARCH(1,1) model using our implementation
logger.info("Fitting GARCH(1,1) model...")
results = fit_garch(returns['portfolio'].ravel()*100)



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
std_resid = pd.Series(
    results.resid / np.sqrt(results.conditional_volatility),
    index=returns.index,
    name="Residual"
)

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
print(forecast_plot)

# %% [markdown]
# # Conclusion
#
# This implementation demonstrates the key components of GARCH modeling:
#
# 1. Data preparation and log returns calculation
# 2. GARCH(1,1) model estimation
# 3. Volatility analysis and visualization
# 4. Model diagnostics
# 5. Volatility forecasting
#
# The GARCH model provides a powerful framework for modeling and forecasting financial volatility, which is essential for risk management and asset pricing.

# %%
# %% [markdown]
# ## Extension: Alternative Volatility Specifications
# We fit three additional conditional‐variance models—EGARCH(1,1), GJR‐GARCH(1,1) and TARCH(1,1)—on the portfolio returns, extract their parameters side by side and plot their implied volatilities.

# %%
# | label: fit‐variants
import arch

logger.info("Fitting alternative GARCH variants…")

models = {
    'GARCH(1,1)': arch.arch_model(returns['portfolio'] * 100, vol='GARCH', p=1, q=1, dist='normal'),
    'EGARCH(1,1)': arch.arch_model(returns['portfolio'] * 100, vol='EGARCH', p=1, q=1, dist='t'),
    'GJR‐GARCH(1,1)': arch.arch_model(returns['portfolio'] * 100, vol='GARCH', p=1, o=1, q=1, power=2.0, dist='normal'),
}
fitted = {'GARCH(1,1)': results}
param_frames = []
for name, res in fitted.items():
    df = res.params.rename(name).to_frame().T
    param_frames.append(df)
comparison = pd.concat(param_frames)
print(comparison.round(4))

for name, model in models.items():
    fitted[name] = model.fit(disp='off')
    logger.info(f"{name} fitted; AIC={fitted[name].aic:.2f}")

# %% [markdown]
# ### Parameter Comparison Table
# The table below lines up the ω, ARCH and GARCH coefficients (and leverage term for GJR) for each model.

# %%
# | label: compare‐variant‐params
param_frames = []
for name, res in fitted.items():
    df = res.params.rename(name).to_frame().T
    param_frames.append(df)
comparison = pd.concat(param_frames)
print(comparison.round(4))

# %% [markdown]
# ## Overlay of Conditional Volatilities
# We compute the in‐sample conditional standard deviations for each model and plot them together to see how the different dynamics evolve over time.

# %%
# | label: plot‐variant‐vols
vol_series = {}
for name, res in fitted.items():
    vol_series[name] = np.sqrt(res.conditional_volatility) / 100  # back to decimal returns

vol_df = pd.DataFrame(vol_series, index=returns.index).reset_index().melt(
    id_vars='Date', var_name='Model', value_name='Volatility'
)

from plotnine import geom_line

variant_vol_plot = (
    ggplot(vol_df, aes(x='Date', y='Volatility', color='Model'))
    + geom_line()
    + labs(title='Conditional Volatility: GARCH vs. EGARCH vs. GJR', x='Date', y='σₜ')
    + theme_minimal()
    + theme(
        plot_title=element_text(size=14, face='bold'),
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
    )
)

print(variant_vol_plot)

# %% [markdown]
# ## Next Steps

# %%

import numpy as np
import pandas as pd
from utils import compute_var, backtest_var


# significance
alpha = 0.01

# container for results
bt_summary = []

# assume `fitted` is your dict of fitted models, and `returns` holds your portfolio returns
for name, res in fitted.items():
    # 1) extract conditional volatility (decimal returns)
    cond_vol = np.sqrt(res.conditional_volatility) / 100  

    # 2) compute 1% VaR series
    var_ser = compute_var(cond_vol, alpha=alpha)

    # 3) backtest: count exceptions
    ret = returns["portfolio"].values  # decimal returns
    bt = backtest_var(ret, var_ser, alpha=alpha)

    # 4) collect summary
    bt_summary.append({
        "Model": name,
        "Exceedances": bt["n_exceed"],
        "Expected": bt["expected_exceed"],
        "Hit Rate": bt["hit_rate"]
    })

# 5) display as DataFrame
bt_df = pd.DataFrame(bt_summary)
print(bt_df.to_string(index=False))
#%%