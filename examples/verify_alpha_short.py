#!/usr/bin/env python3
import datetime
import os
import sys
from pathlib import Path

# --- SDK Import Helper ---
repo_src = (Path(__file__).parent.parent / "src").resolve()
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

from aipricepatterns import Client


def find_alpha_short():
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    symbols = ["BTCUSDT", "ETHUSDT"]
    print("--- Searching for Positive Alpha Showcase (Short Range) ---")

    # Last 30 days
    now = datetime.datetime.now(datetime.timezone.utc)
    start_ts = int((now - datetime.timedelta(days=30)).timestamp() * 1000)
    end_ts = int(now.timestamp() * 1000)

    for symbol in symbols:
        print(f"\nScanning {symbol}...")
        try:
            res = client.backtest(
                symbol=symbol,
                interval="1h",
                start_ts=start_ts,
                end_ts=end_ts,
                min_prob=0.55,
                step=12,
                include_stats=True,
            )
            stats = res.get("stats", {})
            ret = stats.get("totalReturnPct", 0)
            bench = stats.get("benchmarkReturnPct", 0)
            alpha = ret - bench
            print(f"  Return: {ret:.2f}% | Bench: {bench:.2f}% | Alpha: {alpha:+.2f}%")
            print(
                f"  Win Rate: {stats.get('winRate', 0):.2f}% | Signals: {stats.get('totalSignals', 0)}"
            )
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    find_alpha_short()
