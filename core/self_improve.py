"""100-trade self-improvement loop.

Tracks completed_trades count; every time it crosses a multiple of 100
(100, 200, 300, ...), triggers a walk-forward retrain that augments the
feature matrix with real-trade performance columns (rolling win rate,
rolling PnL, rolling volatility) and logs metrics + saves model weights.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.features import build_features
from core.labeling import triple_barrier
from core.logger import MODEL_EVOLUTION_HEADERS
from core.model import SignalModel

log = logging.getLogger("quant_bot")

MODELS_DIR = Path("data/models")
DEFAULT_TRIGGER_EVERY = 100

TRADE_FEATURES = [
    "trade_count_30d", "rolling_win_rate", "rolling_avg_pnl", "rolling_volatility",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(v):
    return "" if v is None else v


def _metrics_row(**kwargs) -> dict:
    return {h: _safe(kwargs.get(h)) for h in MODEL_EVOLUTION_HEADERS}


class SelfImprovementLoop:
    """Watches closed trade count and triggers retraining every N trades."""

    def __init__(
        self,
        trade_log_path: Path = Path("data/trades.csv"),
        trigger_every: int = DEFAULT_TRIGGER_EVERY,
        model_dir: Path = MODELS_DIR,
    ):
        self.trade_log_path = Path(trade_log_path)
        self.trigger_every = int(trigger_every)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._last_trigger_at: int = 0

    def completed_trade_count(self) -> int:
        if not self.trade_log_path.exists():
            return 0
        try:
            df = pd.read_csv(self.trade_log_path)
            if df.empty:
                return 0
            return int(len(df))
        except Exception as exc:
            log.warning("Trade log read failed: %s", exc)
            return 0

    def win_count(self) -> int:
        if not self.trade_log_path.exists():
            return 0
        try:
            df = pd.read_csv(self.trade_log_path)
            if df.empty or "pnl_usd" not in df.columns:
                return 0
            return int((df["pnl_usd"] > 0).sum())
        except Exception:
            return 0

    def win_rate(self) -> float:
        total = self.completed_trade_count()
        return 0.0 if total <= 0 else self.win_count() / total

    def should_trigger(self) -> bool:
        n = self.completed_trade_count()
        if n <= 0:
            return False
        threshold = (n // self.trigger_every) * self.trigger_every
        return threshold > self._last_trigger_at and threshold > 0

    def _augment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = build_features(df)
        for col in TRADE_FEATURES:
            feats[col] = np.nan
        if not self.trade_log_path.exists():
            return feats
        try:
            trades = pd.read_csv(self.trade_log_path)
        except Exception:
            return feats
        if trades.empty or "exit_time" not in trades.columns:
            return feats

        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
        trades = trades.dropna(subset=["exit_time"])

        for i in range(len(feats)):
            bar_ts = feats.index[i]
            window_start = bar_ts - pd.Timedelta(days=30)
            window = trades[(trades["exit_time"] >= window_start) & (trades["exit_time"] <= bar_ts)]
            n = len(window)
            feats.iat[i, feats.columns.get_loc("trade_count_30d")] = n
            feats.iat[i, feats.columns.get_loc("rolling_win_rate")] = (
                float((window["pnl_usd"] > 0).mean()) if n > 0 else 0.0
            )
            feats.iat[i, feats.columns.get_loc("rolling_avg_pnl")] = (
                float(window["pnl_usd"].mean()) if n > 0 else 0.0
            )
            feats.iat[i, feats.columns.get_loc("rolling_volatility")] = (
                float(window["pnl_usd"].std()) if n > 1 else 0.0
            )
        return feats

    def retrain(self, df: pd.DataFrame, reason: str = "100-trade milestone") -> tuple[SignalModel, dict]:
        feats = self._augment_features(df)
        labels = triple_barrier(df["high"], df["low"], df["close"], feats["atr"])
        data = feats.join(labels.rename("label")).dropna()
        if len(data) < 120:
            raise RuntimeError(f"Not enough samples after augmentation: {len(data)}")

        feature_cols = list(feats.columns)
        split = int(len(data) * 0.8)
        model = SignalModel().train(data[feature_cols].iloc[:split], data["label"].iloc[:split].astype(int))

        y_test = data["label"].iloc[split:].astype(int).to_numpy()
        y_pred = model.model.predict(data[feature_cols].iloc[split:].to_numpy())
        oos_acc = float((y_pred == y_test).mean())
        y_in = data["label"].iloc[:split].astype(int).to_numpy()
        y_pred_in = model.model.predict(data[feature_cols].iloc[:split].to_numpy())
        in_acc = float((y_pred_in == y_in).mean())

        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_path = self.model_dir / f"model_{timestamp_tag}.joblib"
        try:
            joblib.dump(
                {"estimator": model.model, "feature_columns": feature_cols, "trained_at": _now_iso()},
                model_path,
            )
        except Exception as exc:
            log.warning("Model save failed: %s", exc)
            model_path = ""

        n_trades = self.completed_trade_count()
        threshold = (n_trades // self.trigger_every) * self.trigger_every
        self._last_trigger_at = threshold

        metrics = _metrics_row(
            timestamp=_now_iso(),
            trigger=reason,
            n_features=len(feature_cols),
            n_samples=int(len(data)),
            n_estimators=int(getattr(model.model, "n_estimators", 0) or 0),
            learning_rate=float(getattr(model.model, "learning_rate", 0.0) or 0.0),
            oos_accuracy=oos_acc,
            in_sample_accuracy=in_acc,
            trade_count=n_trades,
            duration_seconds=0.0,
            model_path=str(model_path),
            notes=f"Augmented features: {TRADE_FEATURES}",
        )
        log.info(
            "Self-improve retrain @ %d trades: features=%d samples=%d OOS=%.3f IN=%.3f -> %s",
            n_trades, len(feature_cols), len(data), oos_acc, in_acc, model_path,
        )
        return model, metrics

    def maybe_retrain(self, df: pd.DataFrame, logger=None) -> SignalModel | None:
        if not self.should_trigger():
            return None
        t0 = time.time()
        model, metrics = self.retrain(df, reason=f"Every-{self.trigger_every} trades")
        metrics["duration_seconds"] = round(time.time() - t0, 3)
        if logger is not None:
            try:
                logger.log_model_retrain(metrics)
            except Exception as exc:
                log.warning("Log model retrain failed: %s", exc)
        return model
