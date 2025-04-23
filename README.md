# GARCH Model Implementation

This project implements GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models based on the paper "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics" by Robert F. Engle.

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
- Poetry (for dependency management)
- Quarto (for rendering the Quarto document)
- pre-commit (for code quality checks)
- easy-commit (optional, for writing more meaningful and easier commits)
- vscode / jupyter notebook / jupyterlab (optional, for features such as syncing up .qmd / .md / .ipynb / .py files)
- [makedown](https://github.com/tzador/makedown) - for running this file as a script (`pip install --upgrade makedown`)
## Installation

1. Clone this repository
   ```bash
   git clone https://github.com/kwojdalski/garch_project/
   ```
2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Install Quarto (optional, for rendering the Quarto document):
   - Download from [Quarto website](https://quarto.org/docs/get-started/)
5. Set up pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```
## Makedown

This project uses [makedown](https://github.com/tzador/makedown) to run commands directly from the README.md file. This allows you to execute the installation and usage commands directly from this documentation.

### Automated Installation with Makedown

To install automatically, run `makedown function`. For instance:
```bash
makedown install
```

If you have makedown installed, you can run the following commands directly from this README:
## [install]() Installation of the project

It might take a while (few minutes) to install the whole project...

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install -v
poetry run pre-commit install --all-files
pipx install quarto-cli
```

## [render_html]() Render HTML page

Moreover it opens the rendered page.

```bash
quarto render notebooks/garch.qmd &&  open notebooks/garch.html
```


## Usage

### Running the Analysis

The main analysis can be run in several ways:

1. Using the Python script:
   ```bash
   poetry run python src/garch.py
   ```

2. Using the Jupyter Notebook:
   ```bash
   poetry run jupyter notebook notebooks/garch.ipynb
   ```

3. Using the Quarto document:
   ```bash
   poetry run quarto render notebooks/garch.qmd
   ```

### Using the Quarto Presentation

1. Render the Quarto presentation:
   ```bash
   poetry run quarto render notebooks/garch_presentation.qmd
   ```
2. Open the generated HTML file in your browser
3. Use arrow keys or space to navigate through the slides

### Code Quality

This project uses pre-commit hooks to ensure code quality. The hooks are automatically run when you commit changes, but you can also run them manually:

```bash
poetry run pre-commit run --all-files
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
poetry run easy-commit "Your commit message"
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
