from __future__ import annotations

import os
from datetime import datetime

from aipricepatterns import Client


def _fmt_dt(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    client = Client(base_url=base_url, api_key=api_key)

    print("=" * 78)
    print("DEMO 08 — LIVE SIGNALS FEED")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print("Fetching high-probability signals from the background scanner...")

    res = client.get_signals()
    signals = res.get("signals") or []

    print(f"Signals found: {len(signals)}")
    print("-" * 78)

    if signals:
        header = f"{'SYMBOL':10s} {'INT':4s} {'DIR':5s} {'PROB':6s} {'SIM':6s} {'TS'}"
        print(header)
        print("-" * 78)
        for s in signals:
            sym = s.get("symbol", "n/a")
            interval = s.get("interval", "n/a")
            direction = s.get("direction", "n/a")
            prob = s.get("up_prob")
            prob_s = f"{prob:.2f}" if isinstance(prob, (int, float)) else "n/a"
            sim = s.get("similarity")
            sim_s = f"{sim:.2f}" if isinstance(sim, (int, float)) else "n/a"
            ts = s.get("ts")
            dt = _fmt_dt(ts) if isinstance(ts, int) else "n/a"
            print(f"{sym:10s} {interval:4s} {direction:5s} {prob_s:6s} {sim_s:6s} {dt}")
    else:
        print("No active signals found. Try adjusting scanner settings in the API.")

    print("-" * 78)
    print("Traders use this feed to discover opportunities across all indexed assets")
    print("without having to manually search each symbol.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
