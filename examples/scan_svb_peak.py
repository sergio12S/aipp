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


def scan_svb():
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    # We'll check the period where BTC bottomed: March 10-12, 2023
    start_dt = datetime.datetime(2023, 3, 11, 0, 0, 0, tzinfo=datetime.timezone.utc)

    print("--- Scanning SVB Bottom for High Prob Signals ---")
    for h in range(0, 48, 4):  # Check every 4 hours for 2 days
        test_dt = start_dt + datetime.timedelta(hours=h)
        ts = int(test_dt.timestamp() * 1000)

        try:
            data = client.get_pattern_metrics(
                symbol="BTCUSDT",
                interval="1h",
                window_end_ts=ts,
            )
            metrics = data.get("metrics", {})
            up_prob = metrics.get("upProbPct", 0)
            if up_prob > 55:
                print(f"[{test_dt}] Win Rate: {up_prob:.2f}% ✅")
            else:
                print(f"[{test_dt}] Win Rate: {up_prob:.2f}%")
        except:
            pass


if __name__ == "__main__":
    scan_svb()
