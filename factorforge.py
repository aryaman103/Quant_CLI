import argparse
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import os
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the CSV data.
    
    Args:
        file_path: Path to the CSV file containing OHLCV data
        
    Returns:
        DataFrame with parsed dates as index and validated columns
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing or data is empty
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            parse_dates=['date'],
            index_col='date'
        )
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for empty dataframe
        if df.empty:
            raise ValueError("Data file is empty")
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def engineer_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers technical analysis features for the model.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional factor columns:
        - momentum_10d: 10-day momentum
        - sma_gap_20d: Gap from 20-day simple moving average
        - atr_14d: 14-day Average True Range
        - earnings_yield_proxy: Inverse of close price
    """
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['sma_gap_20d'] = df['close'] / df['close'].rolling(20).mean() - 1
    df['atr_14d'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()
    df['earnings_yield_proxy'] = 1 / df['close']
    return df

def run_backtest(df_test: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Runs a simple weekly-rebalanced backtest.
    
    Args:
        df_test: Test DataFrame with OHLCV data and target returns
        predictions: Model predictions for next-day returns
        
    Returns:
        DataFrame with backtest results including equity curve and signals
    """
    df_test = df_test.copy()
    # Force index to be DatetimeIndex
    df_test.index = pd.to_datetime(df_test.index)
    df_test['prediction'] = predictions
    df_test['signal'] = np.where(df_test['prediction'] > 0, 1, 0)
    
    # Rebalance weekly (hold position for a week)
    weekly_signal = df_test['signal'].resample('W-FRI').ffill()
    df_test['signal'] = weekly_signal.reindex(df_test.index, method='ffill').fillna(0)

    df_test['strategy_return'] = df_test['signal'] * df_test['target']
    df_test['equity_curve'] = (1 + df_test['strategy_return'].fillna(0)).cumprod()
    return df_test

def calculate_metrics(equity_curve: pd.Series) -> Tuple[float, float, float]:
    """Calculates performance metrics from equity curve.
    
    Args:
        equity_curve: Time series of cumulative returns
        
    Returns:
        Tuple of (CAGR, Sharpe ratio, Max drawdown)
    """
    total_return = equity_curve.iloc[-1] - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    cagr = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
    
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() != 0 else 0
    )
    
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return cagr, sharpe_ratio, max_drawdown

def plot_equity_curve(equity_curve: pd.Series, output_path: str) -> None:
    """Saves the equity curve plot to file.
    
    Args:
        equity_curve: Time series of cumulative returns
        output_path: Path where the plot image will be saved
    """
    plt.figure(figsize=(10, 6))
    equity_curve.plot(title='Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nEquity curve saved to {output_path}")

def main() -> None:
    """Main function to run the factor engineering and backtesting CLI.
    
    Orchestrates the entire workflow:
    1. Load and validate data
    2. Engineer technical factors
    3. Train ML model with hyperparameter optimization
    4. Run backtest simulation
    5. Calculate and display performance metrics
    6. Generate equity curve visualization
    """
    parser = argparse.ArgumentParser(description="FactorForge CLI Tool")
    parser.add_argument(
        '--csv',
        type=str,
        default='data/sample.csv',
        help='Path to the daily OHLCV CSV file.'
    )
    args = parser.parse_args()

    print("1. Loading data...")
    df = load_data(args.csv)

    print("2. Engineering factors...")
    df = engineer_factors(df)

    # Clean data and prepare features/target
    features = ['momentum_10d', 'sma_gap_20d', 'atr_14d', 'earnings_yield_proxy']
    df['target'] = df['close'].pct_change().shift(-1)
    df_clean = df.dropna()

    # Split entire DataFrame to preserve index
    train_size = int(len(df_clean) * 0.7)
    df_train = df_clean.iloc[:train_size]
    df_test = df_clean.iloc[train_size:]

    X_train, y_train = df_train[features], df_train['target']
    X_test, y_test = df_test[features], df_test['target']

    print("3. Building and training model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        'gbr__n_estimators': [50, 100],
        'gbr__learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")

    print("4. Running backtest...")
    predictions = grid_search.predict(X_test)
    backtest_results = run_backtest(df_test, predictions)

    print("5. Calculating performance metrics...")
    cagr, sharpe, max_dd = calculate_metrics(backtest_results['equity_curve'])
    
    print("\n--- Backtest Metrics ---")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    
    plot_equity_curve(
        backtest_results['equity_curve'], 'reports/equity_curve.png'
    )

if __name__ == "__main__":
    main() 