from __future__ import annotations

import os
import time

from aipricepatterns import BacktestAuditor, Client


def _period_start_ms(days: int) -> int:
    return int((time.time() - days * 24 * 60 * 60) * 1000)


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    client = Client(base_url=base_url, api_key=api_key)

    symbol = os.getenv("AIPP_DEMO_SYMBOL", "BTCUSDT")
    interval = os.getenv("AIPP_DEMO_INTERVAL", "1h")

    days = int(os.getenv("AIPP_DEMO_DAYS", "90"))
    start_ts = _period_start_ms(days)

    q = int(os.getenv("AIPP_DEMO_Q", "60"))
    f = int(os.getenv("AIPP_DEMO_F", "24"))
    step = int(os.getenv("AIPP_DEMO_STEP", "24"))
    min_prob = float(os.getenv("AIPP_DEMO_MIN_PROB", "0.50"))

    bt = client.backtest(
        symbol=symbol,
        interval=interval,
        q=q,
        f=f,
        step=step,
        top_k=5,
        min_prob=min_prob,
        start_ts=start_ts,
    )

    trades = bt.get("results") or bt.get("trades") or []
    trades = trades if isinstance(trades, list) else []

    returns = [t.get("actualReturnPct", 0.0) for t in trades if isinstance(t, dict)]
    losses = [r for r in returns if isinstance(r, (int, float)) and r < 0]

    print("=" * 78)
    print("DEMO 02 — BACKTEST + AUDIT")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(f"Symbol: {symbol}  Interval: {interval}  Days: {days}")
    print(f"Signals: {len(trades)}  Losing signals: {len(losses)}")

    trades_for_audit = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = t.get("ts")
        if not isinstance(ts, int):
            continue
        trades_for_audit.append(
            {"entryTime": ts, "returnPct": t.get("actualReturnPct", 0.0)}
        )

    auditor = BacktestAuditor(client)
    report = auditor.analyze_losses(symbol, interval, trades_for_audit)

    total_losses = report.get("total_losses", 0)
    dist = report.get("regime_distribution", {}) or {}

    print("-")
    print(f"Total losses analyzed: {total_losses}")
    if isinstance(dist, dict) and dist:
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        print("Losses by regime:")
        for rid, cnt in items[:10]:
            pct = (cnt / total_losses * 100) if total_losses else 0
            print(f"  {rid:20s}: {cnt:3d} ({pct:4.1f}%)")
        print(f"Avoid trading in: {items[0][0]}")
    else:
        print("No losing trades to attribute (in this sample).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
