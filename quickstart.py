"""One-command run: fetch data -> build features -> label -> train -> print live signal."""

from __future__ import annotations

import argparse

from core.features import FEATURE_COLUMNS, build_features, load_ohlcv
from core.labeling import triple_barrier
from core.model import SignalModel
from core.telemetry import Telemetry, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTC-USD", help="ticker (yfinance) or market id (ccxt)")
    p.add_argument("--source", choices=["yfinance", "ccxt"], default="yfinance")
    p.add_argument("--interval", default="1d", help="yfinance interval (1h, 1d, ...)")
    p.add_argument("--limit", type=int, default=750, help="bars to fetch")
    p.add_argument("--exchange", default="binance", help="ccxt exchange id when --source ccxt")
    p.add_argument("--timeframe", default="1h", help="ccxt timeframe when --source ccxt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    df = load_ohlcv(
        args.symbol, source=args.source, interval=args.interval, limit=args.limit,
        exchange=args.exchange, timeframe=args.timeframe,
    )
    feats = build_features(df)
    labels = triple_barrier(df["high"], df["low"], df["close"], feats["atr"])

    data = feats.join(labels.rename("label")).dropna()
    if len(data) < 120:
        raise SystemExit(f"Only {len(data)} labeled rows — need >=120 for a meaningful train/test split.")

    X, y = data[FEATURE_COLUMNS], data["label"].astype(int)
    split = int(len(data) * 0.8)
    model = SignalModel().train(X.iloc[:split], y.iloc[:split])

    oos_acc = float((model.model.predict(X.iloc[split:]) == y.iloc[split:].to_numpy()).mean())
    dist = y.value_counts().sort_index()

    last = df.iloc[-1]
    sig = model.signal(
        X.iloc[[-1]].to_numpy(),
        price=float(last["close"]),
        atr=float(data["atr"].iloc[-1]),
    )
    Telemetry(symbol=args.symbol).log_signal(sig)

    print("\n============== SUMMARY ==============")
    print(f"Symbol       : {args.symbol}  ({args.source} / {args.interval or args.timeframe})")
    print(f"Bars         : {len(df)} fetched, {len(data)} labeled (post warmup/dropna)")
    print(f"Label counts : {dist.to_dict()}")
    print(f"Train        : {split} bars | Test: {len(data) - split} bars")
    print(f"OOS accuracy : {oos_acc:.1%}")
    print("-------------------------------------")
    print(f"Live signal  : {sig['signal']}")
    print(f"  P(TP)={sig['p_tp']:.3f}  P(SL)={sig['p_sl']:.3f}  P(expiry)={sig['p_time']:.3f}")
    print(f"  Edge = {sig['edge']:+.4f} (fees+slippage included)")
    print(f"  Entry~{sig['price']:.2f}  TP={sig['tp_price']:.2f}  SL={sig['sl_price']:.2f}")
    print("=====================================")


if __name__ == "__main__":
    main()
