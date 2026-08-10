"""Centralized configuration for the quantitative research pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration object imported by all src/ modules."""

    # Triple Barrier Parameters
    pt_mult: float = 1.5      # TP barrier = close + pt_mult * ATR
    sl_mult: float = 1.0      # SL barrier = close - sl_mult * ATR
    max_hold: int = 5         # Vertical barrier (bars to hold)

    # Validation Parameters
    embargo_pct: float = 0.01  # Purged K-Fold embargo fraction
    n_outer_splits: int = 5    # Outer CV splits for OOS evaluation
    n_inner_splits: int = 3    # Inner CV splits for hyperparameter tuning

    # Risk & Sizing
    kelly_fraction: float = 0.25   # Fractional Kelly (0.25 = quarter-Kelly)
    kelly_cap: float = 0.05        # Max position size as fraction of capital
    min_notional: float = 5.0      # Minimum order notional in USD
    step_size: float = 0.0001      # Position size precision
    drawdown_limit: float = 0.03   # Daily drawdown circuit breaker (3%)
    max_concurrent_exposure: float = 0.5  # Max total exposure across positions

    # Fees & Slippage
    maker_fee: float = 0.0002      # Maker fee (0.02%)
    taker_fee: float = 0.0005      # Taker fee (0.05%)
    slippage_k: float = 0.0001     # Square-root market impact coefficient
    default_slippage_bps: float = 1.0  # Default slippage in basis points

    # Feature Engineering
    atr_period: int = 14           # ATR lookback period
    rsi_period: int = 14           # RSI lookback period
    vol_lookback: int = 20         # Volatility lookback period
    frac_diff_d: float = 0.5       # Fractional differentiation degree

    # Model Parameters
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31

    # Backtest Parameters
    initial_capital: float = 10_000.0
    monte_carlo_iterations: int = 10_000
    block_bootstrap_block_len: int = 10  # Average holding duration in bars

    # Drift Detection
    psi_threshold: float = 0.25    # Population Stability Index threshold
    concept_drift_window: int = 100  # Window for drift detection

    # Regime Filter
    atr_percentile_low: float = 0.2   # Low volatility regime threshold
    atr_percentile_high: float = 0.8  # High volatility regime threshold

    # Data Settings
    default_symbol: str = "BTC/USDT"
    default_timeframe: str = "5m"
    default_limit: int = 500
    train_window: int = 1000        # Rolling window for training
    retrain_every: int = 100        # Retrain after this many new bars

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("data/models"))

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# Global singleton config instance
config = Config()
