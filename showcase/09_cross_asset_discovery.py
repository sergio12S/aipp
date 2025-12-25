from __future__ import annotations

import os

from aipricepatterns import Client


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    client = Client(base_url=base_url, api_key=api_key)

    symbol = os.getenv("AIPP_DEMO_SYMBOL", "BTCUSDT")
    interval = os.getenv("AIPP_DEMO_INTERVAL", "1h")

    print("=" * 78)
    print("DEMO 09 — CROSS-ASSET DISCOVERY")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(f"Query Asset: {symbol} ({interval})")
    print("Searching for similar patterns across ALL indexed assets...")

    # Enable cross_asset=True to find analogues in other symbols
    res = client.search(
        symbol=symbol, interval=interval, q=60, f=24, top_k=10, cross_asset=True
    )

    matches = res.get("matches") or []
    print(f"Found {len(matches)} historical analogues across the market.")
    print("-" * 78)

    if matches:
        print(f"{'#':2s} {'SYMBOL':10s} {'SIMILARITY':12s} {'DATE/TS'}")
        print("-" * 78)
        for i, m in enumerate(matches, start=1):
            msym = m.get("symbol", "n/a")
            sim = m.get("similarity")
            sim_s = f"{sim * 100:.2f}%" if isinstance(sim, (int, float)) else "n/a"
            dt = m.get("date") or m.get("ts") or "n/a"
            print(f"{i:<2d} {msym:10s} {sim_s:12s} {dt}")
    else:
        print("No matches found.")

    print("-" * 78)
    print("Cross-asset discovery allows traders to find 'Market Memory' signatures")
    print("that appear consistently across different asset classes (e.g., BTC vs ETH).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
