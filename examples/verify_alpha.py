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


def find_alpha():
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    print("--- Searching for Positive Alpha Showcase (2024) ---")

    start_2024 = int(datetime.datetime(2024, 1, 1).timestamp() * 1000)
    end_now = int(datetime.datetime.now().timestamp() * 1000)

    for symbol in symbols:
        print(f"\nScanning {symbol}...")
        try:
            # Try with a probability filter
            res = client.backtest(
                symbol=symbol,
                interval="1h",
                start_ts=start_2024,
                end_ts=end_now,
                min_prob=0.55,
                step=24,
                include_stats=True,
            )
            stats = res.get("stats", {})
            ret = stats.get("totalReturnPct", 0)
            bench = stats.get("benchmarkReturnPct", 0)
            alpha = ret - bench
            print(f"  Return: {ret:.2f}% | Bench: {bench:.2f}% | Alpha: {alpha:+.2f}%")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    find_alpha()
