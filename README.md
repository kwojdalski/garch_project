# GARCH Model Implementation

This project implements GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models based on the paper "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics" by Robert F. Engle.

## Project Structure

```
.
├── src/
│   └── garch_implementation.py    # Main GARCH implementation script
├── notebooks/
│   ├── garch_analysis.ipynb      # Jupyter notebook with detailed analysis
│   └── garch_quarto.qmd          # Quarto document with interactive analysis
├── papers/
│   └── Engle-GARCH101Use-2001.pdf # The original research paper
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Features

- GARCH(1,1) model implementation
- Financial data fetching using Yahoo Finance
- Volatility forecasting
- Model diagnostics and visualization
- Interactive analysis through Jupyter notebook and Quarto document

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt
- Quarto (for rendering the Quarto document)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Quarto (optional, for rendering the Quarto document):
   - Download from [Quarto website](https://quarto.org/docs/get-started/)

## Usage

### Running the Main Script

```bash
python src/garch_implementation.py
```

This will:
- Fetch S&P 500 data
- Fit a GARCH(1,1) model
- Generate volatility forecasts
- Save a volatility plot

### Using the Jupyter Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/garch_analysis.ipynb`
3. Run the cells to see detailed analysis and visualizations

### Using the Quarto Document

1. Render the Quarto document:
   ```bash
   quarto render notebooks/garch_quarto.qmd
   ```
2. Open the generated HTML file in your browser

## Model Description

The GARCH(1,1) model is specified as:
$$\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

where:
- $\sigma_t^2$ is the conditional variance
- $\omega$ is the constant term
- $\alpha_1$ is the ARCH effect
- $\beta_1$ is the GARCH effect
- $\varepsilon_{t-1}^2$ is the squared lagged returns
- $\sigma_{t-1}^2$ is the lagged conditional variance

## References

- Engle, R. F. (2001). GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity 
