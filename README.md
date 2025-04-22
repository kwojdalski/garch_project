# GARCH Model Implementation

This project implements GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models based on the paper "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics" by Robert F. Engle.

## Project Structure

```
.
├── src/
│   ├── __init__.py            # Package initialization
│   ├── garch.py        # Main GARCH implementation and analysis
│   └── utils.py               # Utility functions for data processing
├── notebooks/
│   ├── garch.qmd       # Quarto document with interactive analysis
│   ├── garch.ipynb     # Jupyter notebook version of the analysis
│   ├── garch_presentation.qmd # Quarto presentation on GARCH models
│   ├── garch_presentation.html # Rendered presentation
│   └── custom.css             # Custom CSS for the presentation
├── data/                      # Directory for storing financial data
├── papers/
│   └── Engle-GARCH101Use-2001.pdf # The original research paper
├── requirements.txt           # Project dependencies
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── pyproject.toml            # Tool configuration
├── .python-version           # Python version specification
└── README.md                 # This file
```

## Features

- GARCH(1,1) model implementation using the ARCH package
- Financial data fetching using Yahoo Finance
- Volatility forecasting and analysis
- Interactive analysis through Jupyter notebook and Quarto document
- Educational presentation on GARCH models
- Code quality enforcement with pre-commit hooks
- Comprehensive data processing utilities
- Type hints and static type checking with mypy

## Requirements

- Python 3.10+ (specified in .python-version)
- Required packages listed in requirements.txt
- Quarto (for rendering the Quarto document)
- pre-commit (for code quality checks)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Quarto (optional, for rendering the Quarto document):
   - Download from [Quarto website](https://quarto.org/docs/get-started/)
5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Usage

### Running the Analysis

The main analysis can be run in several ways:

1. Using the Python script:
   ```bash
   python src/garch_quarto.py
   ```

2. Using the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/garch_quarto.ipynb
   ```

3. Using the Quarto document:
   ```bash
   quarto render notebooks/garch_quarto.qmd
   ```

### Using the Quarto Presentation

1. Render the Quarto presentation:
   ```bash
   quarto render notebooks/garch_presentation.qmd
   ```
2. Open the generated HTML file in your browser
3. Use arrow keys or space to navigate through the slides

### Code Quality

This project uses pre-commit hooks to ensure code quality. The hooks are automatically run when you commit changes, but you can also run them manually:

```bash
pre-commit run --all-files
```

The following hooks are configured:
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **check-yaml**: Validates YAML files
- **black**: Formats Python code
- **isort**: Sorts Python imports
- **flake8**: Lints Python code
- **mypy**: Type checks Python code
- **bandit**: Checks for security issues

For quick commits with automatic formatting, you can use [easy-commit](https://pypi.org/project/easy-commit/):

```bash
easy-commit "Your commit message"
```

### VS Code Tasks

This project includes VS Code tasks for common development workflows. To use them:

1. Open the project in VS Code
2. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Tasks: Run Task" and select from:
   - `Jupytext Sync`: Sync `.py`, `.qmd`, and `.ipynb` files

You can also use the keyboard shortcut `Cmd+Shift+B` (macOS) or `Ctrl+Shift+B` (Windows/Linux) to run the default build task.

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
