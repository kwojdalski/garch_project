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
"""

# %%
# | label: fetch-data
# Parameters
# 
# we skip fetching, since data is already downloaded
symbols = ["^IXIC", "^DJI", "^TNX"]  # Nasdaq, Dow Jones, and 10-year Treasury
# Define the date range from Robert Engle's paper (2001)
# Sample period from Table 2 and Table 3: March 23, 1990 to March 23, 2000
start_date = datetime(1990, 3, 22)
end_date = datetime(2000, 3, 24)


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

file_path = "data/prices.pkl"
prices.to_pickle(file_path)
"""
# Apparently, DJIA data is missing for 1990-1992, so we'll drop it
## prices = prices.drop(columns=["^DJI"])
## prices = prices.drop(columns=["^TNX"])
# ... we have to take it from somewhere else
# found it here: https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data
# For RATE which looks we'll use
# https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DGS10&scale=left&cosd=2020-04-17&coed=2025-04-17&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=liin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-04-22&revision_date=2025-04-22&nd=1962-01-02

#%%
prices = pd.read_pickle("/Users/shah/garch_project/data/prices.pkl")
print(prices.columns)
#%%
prices = prices.drop(columns= "^DJI")
prices.columns

# %%
djia_prices = pd.read_csv(
    "/Users/shah/garch_project/data/dja-performance-report-daily.csv",
    index_col="dt",
    parse_dates=True,
)
djia_prices.index
#%%
djia_prices = djia_prices.rename(columns={"djia": "^DJI"})
prices = prices.join(djia_prices["^DJI"])
#%%
prices.shape
# %%
# Portfolio weights
weights = {
    "^IXIC": 0.50,  # Nasdaq
    "^DJI": 0.30,  # Dow Jones
    "^TNX": 0.20,  # 10-year Treasury
}
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
garch_results = fit_garch(returns['portfolio']*1000)


# Display model summary
logger.info("\nGARCH Model Summary:")
logger.info(garch_results.summary())
#%%
# | label: fit-arch
# Fit ARCH(1,1) model using our implementation
from src.utils import (fit_arch)
logger.info("Fitting ARCH(1) model...")
arch_results = fit_arch(returns['portfolio']* 1000, p=1)

# Display model summary
logger.info("\nARCH Model Summary:")
logger.info(arch_results.summary())

#%% [markdown]
#%% [markdown]
#%% [markdown]
# *Table 2*  
# **ARCH(1)**
# <hr/>
# <div align="center"><em>Variance Equation</em></div>
# <hr/>
#
# | Variable    |    Coef     |   St. Err    |   Z-Stat   |    P-Value     |
# |------------:|------------:|-------------:|-----------:|---------------:|
# | **C (ω)**      |   3.375518  |     0.187    |   18.025   | 1.251×10⁻⁷²     |
# | **ARCH(1)**    |   0.082901  |     0.03052  |    2.716   | 6.599×10⁻³      |
#
# _Covariance estimator: robust_
#
# *Table 3*  
# **GARCH(1,1)**
# <hr/>
# <div align="center"><em>Variance Equation</em></div>
# <hr/>
#
# | Variable      |     Coef      |   St. Err    |   Z-Stat   |    P-Value     |
# |--------------:|--------------:|-------------:|-----------:|---------------:|
# | **C (ω)**        |   3.375460   |     0.187    |   18.025   | 1.245×10⁻⁷²     |
# | **ARCH(1)**      |   0.082900   |     0.03052  |    2.716   | 6.598×10⁻³      |
# | **GARCH(1)**     |   {β₁_coef}  |  {β₁_std_err}| {β₁_z}    |  {β₁_p}         |
#
# _Covariance estimator: robust_


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
# Extract parameters ARCH
params = arch_results.params
print("ARCH Model Parameters:")
for param, value in params.items():
    logger.info(f"{param}: {value:.6f}")
# Extract parameters GARCH
params = garch_results.params
for param, value in params.items():
    logger.info(f"{param}: {value:.6f}")
print("GARCH Model Parameters:")

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
volatility_plot = plot_volatility(garch_results, returns)

# %% [markdown]
# ## Model Diagnostics
#
# Let's examine the model residuals using plotnine:

# %%
# | label: model-diagnostics
# | fig-cap: Standardized Residuals and Q-Q Plot

# Get standardized residuals
std_resid = pd.Series(
    garch_results.resid / np.sqrt(garch_results.conditional_volatility),
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
forecast = garch_results.forecast(horizon=5)
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
        "Volatility": np.sqrt(garch_results.conditional_volatility[-100:]),
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
# ## Extensions 
# ## Alternative error-distribution specifications
# GARCH(1,1) model estimation (Normal, Student-t, Skew-t, GED)
# Volatility analysis and visualization
# Model diagnostics
# Volatility forecasting

# %%
# %% [markdown]
# ## 6  Alternative Error-Distribution Specifications
# Fit the same GARCH(1,1) under Normal, Student-t, Skew-t, and GED errors,
# compare information criteria, and overlay the conditional-volatility paths.

# %% 
import arch
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_line, labs, theme_minimal, theme, element_text, scale_color_brewer
)
# %%
# 6.1   Estimate GARCH(1,1) with four innovation distributions
distros = {
    "Normal"   : "normal",
    "Student-t": "t",
    "Skew-t"   : "skewt",
    "GED"      : "ged",
}

alt_models = {}
for name, dist in distros.items():
    logger.info(f"Fitting GARCH(1,1) with {name} errors …")
    am = arch.arch_model(
        returns["portfolio"] * 1000,
        vol="GARCH",
        p=1,
        q=1,
        dist=dist,
        rescale=False,
    )
    res = am.fit(disp="off")
    alt_models[name] = res
# %%
# 6.2   Information-criterion comparison table
comp = pd.DataFrame(
    {
        "Distribution": list(alt_models.keys()),
        "LogLik"      : [m.loglikelihood for m in alt_models.values()],
        "AIC"         : [m.aic            for m in alt_models.values()],
        "BIC"         : [m.bic            for m in alt_models.values()],
    }
).sort_values("AIC")
logger.info("\nError-Distribution Comparison:\n%s", comp.to_string(index=False))

# %%
# 6.3  Overlay conditional-volatility paths

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

for name, res in alt_models.items():
    ax.plot(
        res.conditional_volatility.index,
        np.sqrt(res.conditional_volatility),
        label=name,
        linewidth=1,
    )

ax.set_title("Conditional Volatility – Alternative Error Distributions")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# %%
# 6.4   Q-Q plots of standardized residuals
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, res) in zip(axes, alt_models.items()):
    std_resid = res.resid / np.sqrt(res.conditional_volatility)
    if name == "Student-t":
        nu = res.params["nu"]
        stats.probplot(std_resid, dist="t", sparams=(nu,), plot=ax)
    else:  # Normal, Skew-t, GED → compare to N(0,1)
        stats.probplot(std_resid, dist="norm", plot=ax)
    ax.set_title(f"{name} errors")

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Alternative Volatility Specifications

# We fit three additional conditional‐variance models—EGARCH(1,1), GJR‐GARCH(1,1) and TARCH(1,1)—on the portfolio returns, extract their parameters side by side and plot their implied volatilities.
# *Table 1*  
# **EGARCH(1,1)**
# <hr/>
# <div align="center"><em>Variance Equation</em></div>
# <hr/>
#
# | Variable    |    Coef     |   St. Err    |   Z-Stat   |    P-Value     |
# |------------:|------------:|-------------:|-----------:|---------------:|
# | **alpha[1]**   |   0.095657   |   0.018533   |  5.161418  |    2.45E-07    |
# | **beta[1]**    |   0.984675   |   0.005594   | 176.035053 |    0.00E+00    |
# | **gamma[1]**   |      NaN     |      NaN     |     NaN    |       NaN      |
# | **mu**         |  -0.059259   |   0.030740   |  -1.927751 |    5.39E-02    |
# | **nu**         |   5.520671   |   0.573608   |   9.624466 |    6.30E-22    |
# | **omega**      |   0.023097   |   0.007324   |   3.153660 |    1.61E-03    |
#
# _Covariance estimator: robust_
#
# *Table 2*  
# **GARCH(1,1)**
# <hr/>
# <div align="center"><em>Variance Equation</em></div>
# <hr/>
#
# | Variable    |    Coef     |   St. Err    |   Z-Stat   |    P-Value     |
# |------------:|------------:|-------------:|-----------:|---------------:|
# | **alpha[1]**   |   0.042309   |   0.011004   |   3.844999 |    0.000121    |
# | **beta[1]**    |   0.940358   |   0.015847   |  59.338601 |       0        |
# | **gamma[1]**   |      NaN     |      NaN     |     NaN    |       NaN      |
# | **mu**         |  -0.055487   |   0.033495   |  -1.656560 |    0.097608    |
# | **nu**         |      NaN     |      NaN     |     NaN    |       NaN      |
# | **omega**      |   0.058653   |   0.025826   |   2.271130 |    0.023139    |
#
# _Covariance estimator: robust_
#
# *Table 3*  
# **GJR-GARCH(1,1)**
# <hr/>
# <div align="center"><em>Variance Equation</em></div>
# <hr/>
#
# | Variable    |    Coef     |   St. Err    |   Z-Stat   |    P-Value     |
# |------------:|------------:|-------------:|-----------:|---------------:|
# | **alpha[1]**   |   0.031845   |   0.014090   |   2.260150 |    0.023812    |
# | **beta[1]**    |   0.947663   |   0.016999   |  55.746914 |       0        |
# | **gamma[1]**   |   0.016657   |   0.013657   |   1.219652 |    0.222597    |
# | **mu**         |  -0.062503   |   0.033477   |  -1.867041 |    0.061896    |
# | **nu**         |      NaN     |      NaN     |     NaN    |       NaN      |
# | **omega**      |   0.043767   |   0.026439   |   1.655392 |    0.097845    |
#
# _Covariance estimator: robust_


# %%
# | label: fit‐variants
import arch

logger.info("Fitting alternative GARCH variants…")

models = {
    'GARCH(1,1)': arch.arch_model(returns['portfolio'] * 1000, vol='GARCH', p=1, q=1, dist='normal'),
    'EGARCH(1,1)': arch.arch_model(returns['portfolio'] * 1000, vol='EGARCH', p=1, q=1, dist='t'),
    'GJR‐GARCH(1,1)': arch.arch_model(returns['portfolio'] * 1000, vol='GARCH', p=1, o=1, q=1, power=2.0, dist='normal'),
}
fitted = {'GARCH(1,1)': garch_results}
param_frames = []
for name, res in fitted.items():
    df = res.params.rename(name).to_frame().T
    param_frames.append(df)
comparison = pd.concat(param_frames)
print(comparison.round(4))

for name, model in models.items():
    fitted[name] = model.fit(disp='off')
    logger.info(f"{name} fitted; AIC={fitted[name].aic:.2f}")

import pandas as pd

# gather coef, std err, z‐stat and p‐value from each fitted model
summary_list = []
for name, res in fitted.items():
    df = pd.DataFrame({
        'Coef':    res.params,
        'StdErr':  res.std_err,
        'ZStat':   res.params / res.std_err,
        'PValue':  res.pvalues
    })
    df['Model'] = name
    summary_list.append(df)

# concatenate & pivot into a wide table
summary_df = pd.concat(summary_list).reset_index().rename(columns={'index':'Parameter'})
table = summary_df.pivot(index='Parameter', columns='Model', values=['Coef','StdErr','ZStat','PValue'])

from IPython.display import display

# …after building and rounding `table`…
display(table)
#%%
#%% [markdown]
# *Table X*  
# **Comparison of Volatility Model Parameters**
#
# | Parameter | EGARCH(1,1) Coef | EGARCH(1,1) StdErr | EGARCH(1,1) Z-Stat | EGARCH(1,1) P-Value    | GARCH(1,1) Coef | GARCH(1,1) StdErr | GARCH(1,1) Z-Stat | GARCH(1,1) P-Value | GJR-GARCH(1,1) Coef | GJR-GARCH(1,1) StdErr | GJR-GARCH(1,1) Z-Stat | GJR-GARCH(1,1) P-Value |
# |----------:|-----------------:|--------------------:|------------------:|-----------------------:|----------------:|-------------------:|-----------------:|--------------------:|--------------------:|-----------------------:|----------------------:|-----------------------:|
# | **alpha[1]**  |          0.098045 |             0.018890 |          5.190426 |        0.0000002098139 |          0.042579 |             0.010397 |          4.095326 |            0.000042 |             0.033785 |                0.012970 |             2.604787 |               0.009193 |
# | **beta[1]**   |          0.977176 |             0.007679 |        127.249935 |                 0.000000 |          0.929152 |             0.015959 |         58.220658 |            0.000000 |             0.935615 |                0.015781 |            59.288807 |               0.000000 |
# | **gamma[1]**  |                 — |                    — |                 — |                     — |                 — |                    — |                 — |                     — |             0.016370 |                0.016088 |             1.017477 |               0.308926 |
# | **mu**        |         –0.051296 |             0.036680 |         –1.398500 |              0.161963 |         –0.033124 |             0.039475 |         –0.839097 |            0.401415 |            –0.040661 |                0.039831 |            –1.020846 |               0.307327 |
# | **nu**        |          5.670583 |             0.645615 |          8.783232 | 0.000000000000000001588427 |                 — |                    — |                 — |                     — |                 — |                    — |                 — |                     — |
# | **omega**     |          0.033905 |             0.010549 |          3.213972 |              0.001309126 |          0.103500 |             0.037050 |          2.793485 |            0.005214 |             0.084706 |                0.035097 |             2.413497 |               0.015800 |


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

# %% 

returns.head()
returns['portfolio']

#%% [markdown]
#%% [markdown]
# # VaR Backtesting Results
#
# The following table summarizes the backtest of 1% Value-at-Risk (VaR) forecasts from two volatility models over approximately 5,025 trading days:
#
# | Model       | Exceedances | Expected | Hit Rate |
# |-------------|------------:|---------:|---------:|
# | GARCH(1,1)  |       1,729 |    50.25 |   0.3441 |
# | EGARCH(1,1) |       1,182 |    50.25 |   0.2352 |
#
# - **Exceedances**: Number of days when actual portfolio losses exceeded the 1% VaR threshold.
# - **Expected**: Theoretical number of exceedances under a perfect 1% VaR model (N × 0.01 → 50.25 ≈ 5,025 × 0.01).
# - **Hit Rate**: Proportion of exceedances relative to total observations.
#
# ## Interpretation
#
# 1. **Severe Underestimation of Risk**  
#    - GARCH(1,1) flagged far too few losses: actual hits (~34.4%) vs. target 1%.  
#    - EGARCH(1,1) improves but still overshoots: ~23.5% vs. 1%.
#
# 2. **Model Comparison**  
#    - EGARCH(1,1) moderates extremes better than GARCH(1,1), but remains unreliable for tail risk.
#
# ## Recommendations
#
# - **Adopt Heavier-Tails**: Use Student-t or skewed distributions instead of Normal.  
# - **Refine Volatility Specs**: Include leverage effects, asymmetry terms, or higher (p, q) orders.  
# - **Alternative VaR Techniques**: Consider filtered historical simulation or Extreme Value Theory for tail fitting.  
# - **Statistical Backtests**: Conduct Kupiec’s proportions test and Christoffersen’s conditional coverage test to formally assess model adequacy.
#
# ---
# *Generated by analysis on April 28, 2025*

#%%
import numpy as np
import pandas as pd
from arch import arch_model
from src.utils import compute_var, backtest_var

# significance
alpha = 0.01

#  choosing GARCH variant
fitted = {}
specs = {
    "GARCH(1,1)":  dict(vol="Garch",  p=1, q=1),
    "EGARCH(1,1)": dict(vol="Egarch", p=1, q=1)
}

# 2) FITTING
for name, kwargs in specs.items():
    am = arch_model(
        returns["portfolio"],
        rescale=False,      # disable auto-rescaling
        **kwargs
    )
    fitted[name] = am.fit(
        disp="off",
        show_warning=False, # suppress ConvergenceWarning
        options={"maxiter": 5000}
    )

# 3) now backtest the non-empty `fitted`
bt_summary = []
for name, res in fitted.items():
    cond_vol = np.sqrt(res.conditional_volatility) / 100  
    var_ser  = compute_var(cond_vol, alpha=alpha)
    ret      = returns["portfolio"].values
    bt       = backtest_var(ret, var_ser, alpha=alpha)

    bt_summary.append({
        "Model":       name,
        "Exceedances": bt["n_exceed"],
        "Expected":    bt["expected_exceed"],
        "Hit Rate":    bt["hit_rate"]
    })

# 4) display results
bt_df = pd.DataFrame(bt_summary)
print(bt_df.to_string(index=False))

# %%
