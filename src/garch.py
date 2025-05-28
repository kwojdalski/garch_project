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
# The main goal of this projectis to replicate the GARCH model analysis presented in Robert Engle's 2001 Nobel Prize lecture, "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics." The lecture demonstrates how GARCH models can be used to analyze and forecast financial market volatility. However to make things more interesting we add some extensions to the study. First of all, we provide our own GARCH(1,1) implementation. We compare GARCH(1,1) results with some other volatility models, particularly E-GARCH and GJR-GARCH on the same data sample that was used in the original paper. Finally, we examine how market volatility has evolved since the publication of the paper.
#
# ### Rationale behind the Study
#
# 1) From the initial inspection of the article, easy to obtain data
# 2) Well-known author so we asssumed that the methodology is sound and well-explained
# 3) The author's research has had significant influence and impact in the field of time series analysis
# 4) Research could be easily extended for different datasets, portfolios and methods
#
#
# ### Steps in the Research
# The replication focuses on the following:
#
# 1. Constructing a portfolio similar (or, hopefully, identical) to the one used in Engle's paper
# 2. Calculating and analyzing portfolio returns
# 3. Writing GARCH(1,1) implementation
# 4. Fitting a GARCH(1,1) model to the portfolio returns
# 5. Examining the model's performance in capturing volatility clustering
# 6. Generating volatility forecasts in the same manner as Engle did
# 7. Extending study with other methods
# 8. Examining portfolio built on current data
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

from src.utils import *

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
returns_oos = returns.loc['2000-03-24':]
returns = returns.loc[:'2000-03-23']

prices_oos = prices.loc['2000-03-24':]
prices = prices.loc[:'2000-03-23']

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
# ARCH models were first introduced by Engle in 1982 to model time-varying volatility in economic and financial time series. The idea was as follows - high squared residual at time $t$ leads to increased conditional variance at time $t+p$. We can write such ARCH(p) process:
# $$
# \begin{cases}
# r_t = \mu_t + \varepsilon_t \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_t^2) \\
# \sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \alpha_2 \varepsilon_{t-2}^2 + ... + \alpha_p \varepsilon_{t-p}^2
# \end{cases}
# $$
# However as shocks in time series often tend to be highly persistent, proper ARCH specification required quite big number of lags p. 
#
# This lead to obvious extension that includes past values of conditional variance. Such solution was proposed by Bollerslev in 1986 and is known as GARCH model. Interestingly Bollerslev was a student of Engle, who supervised his doctoral work.
#
# GARCH(p,q) model is defined as follows:
# $$
# \begin{cases}
# r_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_t^2) \\
# \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2
# \end{cases}
# $$
# We estimate model parameters using maximum likelihood estimator. Assuming conditional normality we define likelihood function as follows:
# $$
# L(\theta) = \prod_{t=1}^T \frac{1}{\sqrt{2\pi \sigma_t^2}} \exp\left( -\frac{(r_t - \mu_t)^2}{2 \sigma_t^2} \right)
# $$
#
# ## GARCH(1,1)
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
# We apply procedure to estimate GARCH(1,1) parameters
#

# %%
# Define initial parameters
r = returns['portfolio']*100
mu = np.mean(r)
resid = r - mu
resid_sq = resid**2


# Log likelihood
def negative_loglik(params):
    omega, alpha, beta = params
    sigma2 = np.zeros(len(r))

    # Calculate conditional variance
    sigma2[0] = omega/(1-alpha-beta)

    for t in range(1, len(sigma2)):
        sigma2[t] = omega + alpha*resid_sq[t-1] + beta*sigma2[t-1]

    
    logl = np.log(1/(np.sqrt(2*np.pi*sigma2))*np.exp(-resid_sq/(2*sigma2)))
    sum_logl = np.sum(logl)

    return -sum_logl


# %% [markdown]
# We do not write our own optimizing function as it is out of the scope of this project. Instead we use existing implementation from scipy

# %%
# We want to maximize sum of log likelihoods
from scipy.optimize import minimize

initial_guess = [np.var(r), 0, 0]
bounds = [(0, None), (0, None), (0, None)]

def constraint_stationarity(params):
    _, alpha, beta = params
    return 1 - alpha - beta

constraints = ({
    'type': 'ineq',
    'fun': constraint_stationarity
})

optimal_res = minimize(
    negative_loglik,
    initial_guess,
    bounds=bounds,
    constraints=constraints,
    options={'maxiter' : 100}
)
print('Success:', optimal_res.success)
print('Message:', optimal_res.message)
print('n iterations', optimal_res.nit)

# %%
omega, alpha, beta = optimal_res.x

# %%
from tabulate import tabulate

table = [
    ['omega', omega],
    ['alpha', alpha],
    ['beta', beta]
]

print(tabulate(table, headers=['Parameter', 'Value'], floatfmt=".6f"))

# %%
# Conditional volatility
sigma2 = np.zeros(len(r))

# Calculate conditional variance
sigma2[0] = omega/(1-alpha-beta)

for t in range(1, len(sigma2)):
    sigma2[t] = omega + alpha*resid_sq[t-1] + beta*sigma2[t-1]

# %%
plt.figure(figsize=(12, 6))
plt.plot(np.sqrt((r.values/100)**2), color='black', lw=0.9)
plt.plot(np.sqrt(sigma2/10000), color='red', lw=1.1)


# %% [markdown]
# ## GARCH Model Existing Implementation
#
# Above we have made our own implementation of GARCH(1,1) model. However there are already existing implementations. We will proceed with the study using arch_model from arch library.
#
# #### Article GARCH(1,1) Model (for reference)
# ![GARCH(1,1) Model](../data/screenshots/garch_one_one.png)

# %% [markdown]
# We use arch library to fit GARCH(1, 1) model. However fitting the model to our returns in the original scale produces warning with recommendation to scale the returns by multiplying them by 1000. Therefore we fit the model to scaled returns. Here we have to remember that forecasted conditional volatility obtained from the resulting object will be scaled as well. For visuals we will rescale back to the original scale.

# %%
# | label: tbl-garch-one-one
# | tbl-cap: GARCH(1,1)
# | tbl-cap-location: top

logger.info("Fitting GARCH(1,1) model...")
results = fit_garch(returns["portfolio"]*1000, vol="Garch", dist="normal")

#logger.info(results.summary())
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
# The same as before we generate series of autocorrelations up to 15th lag together with Q-stats and their p-values. Results are pretty similar to those presented in the original paper. Note that here standardized residuals are basically returns adjusted for GARCH effects as mean portfolio return is very close to 0. This way we can interpret the results by saying that GARCH(1, 1) model manages to capture portfolio returns variance variability.
#
# #### Article ACF of Squared Standardized Residuals (for reference)
# ![ACF of Squared Standardized Residuals](../data/screenshots/squared_standardized_residuals.png)

# %%
# | label: tbl-squared-residuals
# | tbl-cap: Autocorrelations of Squared Standardized Residuals
acf_plot = calculate_acf_table(results.resid / results.conditional_volatility)
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
std_resid = (results.resid / np.sqrt(results.conditional_volatility))/1000

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
logger.info(forecast.variance.iloc[-1]/1000)

# %% [markdown]
# ## Visualizing Forecasts
#
# Visual below shows conditional volatility and realized volatility defined as square root of squared returns assuming 0 mean:
# $$
# \text{Var}(r_t) = \mathbb{E}[(r_t - \mu)^2]
# $$
# if $\mu = 0$ we get:
# $$
# \text{Var}(r_t) = \mathbb{E}[r_t^2]
# $$

# %%
# | label: fig-forecast
# | fig-cap: Volatility Forecast

# Plot volatility
plt.figure(figsize = (14, 8))
plt.plot(np.sqrt(returns['portfolio']**2), color = 'black', lw = 0.9)
plt.plot(results.conditional_volatility/1000, color = 'red', lw = 1.1)
plt.grid(alpha = 0.3)
plt.title("Volatility realized vs predicted")
plt.show()


# %% [markdown]
# At this point we can also compare results from our implementation with those obtained using arch library implementation

# %%
plt.figure(figsize = (14, 8))
plt.plot(np.sqrt(returns['portfolio']**2), color = 'black', lw = 0.9, label = 'realized volatility')
plt.plot(r.index, np.sqrt(sigma2/10000), color = 'green', lw = 1.1, label = 'GARCH - our implementation')
plt.plot(results.conditional_volatility/1000, color = 'red', lw = 1.1, label = 'GARCH - arch library')
plt.grid(alpha = 0.3)
plt.title("Volatility realized vs predicted")
plt.show()

# %% [markdown]
# # Value at Risk

# %% [markdown]
# Based on conditional volatility we calculate 1% value at risk. We assume initial portfolio value of 1 000 000$. We predict that 1 step ahead standard deviation will be 0.002487. 
#
# First we assume normal distribution of returns. We obtain 1% quantile by multiplying predicted standard deviation by 2.327. 

# %%
init_value = 10**6
2.327*forecast.variance['h.1']/1000 * init_value

# %% [markdown]
# The 1% Value at Risk asusming normal distribution of standardized residuals is 5786$. Note that as mean return is very close to 0 we can omit this value in our calculation. We basically calculated the 99% quantile, but with 0 mean it is symmetrical.

# %% [markdown]
# We can also use empirical 1% quantile

# %%
emp_q = np.quantile(results.resid / np.sqrt(results.conditional_volatility), 0.01)
-emp_q*forecast.variance['h.1']/1000 * init_value

# %% [markdown]
# Both values are significantly lower than those presented in the original paper. This comes from the discrepancies between our data and data used for the original study. It could be seen on portfolio returns plots that our portfolio returns are generally much lower than returns of portfolio from the study.

# %% [markdown]
# We also plot 1% value at risk for the whole in sample period. For reference we show the same plot from the original study.

# %% [markdown]
# ![in-sample-VaR](../data/screenshots/in_sample_VaR.png)

# %%
# Calculate VaR
VaR_returns = calculate_VaR_returns(returns["portfolio"], results.conditional_volatility/1000, 2.327)

# %%
# | label: fig-portfolio_loss
# | fig-cap: Portfolio loss in sample

# Value at risk - portfolio value 1 000 000$ at each point in time
init_value = 10**6
portfolio_returns = returns['portfolio']*init_value
portfolio_VaR = VaR_returns*init_value

plt.figure(figsize = (14, 8))
plt.plot(-portfolio_returns, color = 'black', lw = 1, label='Loss')
plt.plot(-portfolio_VaR, color = 'red', lw = 0.9, label='VaR')
plt.grid(alpha=0.3)
plt.title("Portfolio loss and 99% Value at Risk in sample")
plt.legend()
plt.show()

# %% [markdown]
# We show also the number of exceedances. Good model should produce such VaR estimates that actual portfolio loss is greater than the estimated around 1% of time. We see that for our GARCH(1, 1) VaR estimates assuming normal distribution of standardized residuals is 1.49%.

# %%
# Exceedances
n_exc, perc_exc = count_exceedances(VaR_returns, returns["portfolio"], True)
print(f"Number of exceedances: {n_exc}")
print(f"Percentge of exceedances: {perc_exc*100:.2f}%")


# %% [markdown]
# # Out-of-sample model fit and Value at Risk

# %% [markdown]
# We obtain oit of sample predictions using rolling forecast without reestimating GARCH parameters

# %% [markdown]
# ![oos-VaR](../data/screenshots/out_of_sample_VaR.png)

# %%
# Rolling forecast without reestimating parameters
mu = results.params['mu']
omega = results.params['omega']
alpha = results.params['alpha[1]']
beta = results.params['beta[1]']

epsilon2 = (returns_oos['portfolio']*1000 - mu)**2
sigma2 = np.zeros(len(returns_oos))
sigma2[0] = omega/(1 - alpha - beta)

for t in range(1, len(sigma2)):
    sigma2[t] = omega + alpha*epsilon2[t-1] + beta*sigma2[t-1]


# %%
plt.figure(figsize = (14, 8))
plt.plot(np.sqrt((returns_oos['portfolio'].values)**2), color = 'black', lw = 1)
plt.plot(np.sqrt(sigma2)/1000, color = 'red', lw = 0.9)
plt.show()

# %%
# Value at risk - portfolio value 1 000 000$ at each point in time
plt.figure(figsize = (14, 8))
portfolio_returns = returns_oos['portfolio'].values*(10**6)
plt.plot(-portfolio_returns, color = 'black', lw = 1)
plt.plot(np.sqrt(sigma2) * 2.327 * 10**3, color = 'red', lw = 0.9)
plt.grid(alpha=0.3)
plt.title("Portfolio loss and 99% Value at Risk")
plt.show()

# %%
n_exc, perc_exc = count_exceedances(np.sqrt(sigma2) * (-2.327)/1000, returns_oos['portfolio'].values, True)
print(f"Number of exceedances: {n_exc}")
print(f"Percentge of exceedances: {perc_exc*100:.2f}%")

# %% [markdown]
# # References
#
# * Engle, R. F. (2001). GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics. Journal of Economic Perspectives, 15(4), 157-168.
# * Kaggle. (2023). Dow Jones Industrial Average (^DJI) Data. [Dataset]. Available at: https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data
# * St. Louis Fed. (2023). 10-Year Treasury Constant Maturity Rate [^TNX]. [Dataset]. Available at: https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DGS10&scale=left&cosd=2020-04-17&coed=2025-04-17&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=liin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-04-22&revision_date=2025-04-22&nd=1962-01-02
# * Yahoo Finance. (2023). [Dataset]. Available at: https://finance.yahoo.com/quote/%5EIXIC/history?period1=631152000&period2=1713878400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
# * GenAI used for proof checking and type hinting
