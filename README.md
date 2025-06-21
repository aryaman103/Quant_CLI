# FactorForge

FactorForge is a self-contained, command-line toolkit designed for the rapid prototyping and evaluation of quantitative trading factors. Built as a foundational demonstration of a typical quant workflow, the project showcases how classical feature engineering and machine learning can be combined to generate and backtest trading signals from simple price data. It uses `pandas` for data manipulation, the `ta` library for technical indicator generation, and `scikit-learn` for building a predictive `GradientBoostingRegressor` model. The entire process—from data loading and factor creation to model training, backtesting, and performance reporting—is encapsulated in a single, easy-to-use Python script. FactorForge exists as a lean, CPU-only laboratory for fundamental quant research, providing a clear and extensible baseline for more complex strategy development.

## Project Structure
```
factorforge/
├── data/           # Contains sample daily bar data (CSV)
├── reports/        # Default output directory for generated charts
├── tests/          # unittest suite for core functions
├── factorforge.py  # The core single-file CLI application
├── README.md       # Project documentation (this file)
└── requirements.txt # Python package dependencies
```

## Key Components
*   **Factor Engineering Engine:** Generates common alpha factors from OHLCV data using `pandas` and `ta`. Includes 10d Momentum, 20d SMA Gap, 14d ATR, and an earnings-yield proxy.
*   **ML Pipeline & Training:** Utilizes an `sklearn` Pipeline to chain a `StandardScaler` with a `GradientBoostingRegressor`. Hyperparameters (`n_estimators`, `learning_rate`) are optimized using `GridSearchCV`.
*   **Backtesting Simulator:** Implements a simple, event-driven backtest. The strategy takes a long position based on the model's predicted next-day return, rebalancing on a weekly basis.
*   **Performance & Visualization:** Calculates standard performance metrics (CAGR, Sharpe Ratio, Max Drawdown) and exports a `matplotlib` equity curve PNG for visual analysis of strategy performance.

## Dev Environment
*   Python 3.9+
*   Dependencies managed via `pip` and `requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python factorforge.py --csv path/to/your/data.csv
```

If no path is provided, it will use `data/sample.csv` by default.

The tool will:
- Generate factors from the price data.
- Train a Gradient Boosting model to predict next-day returns.
- Run a simple backtest on the test data.
- Print performance metrics (CAGR, Sharpe Ratio, Max Drawdown).
- Save an equity curve plot to `reports/equity_curve.png`. 