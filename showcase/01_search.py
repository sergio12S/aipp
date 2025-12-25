from __future__ import annotations

import os

from aipricepatterns import Client


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    client = Client(base_url=base_url, api_key=api_key)

    symbol = os.getenv("AIPP_DEMO_SYMBOL", "BTCUSDT")
    interval = os.getenv("AIPP_DEMO_INTERVAL", "1h")
    cross_asset = os.getenv("AIPP_DEMO_CROSS_ASSET", "false").lower() == "true"

    res = client.search(
        symbol=symbol,
        interval=interval,
        q=60,
        f=24,
        top_k=10,
        cross_asset=cross_asset,
    )

    matches = res.get("matches") or []
    meta = res.get("meta") or {}

    print("=" * 78)
    print("DEMO 01 — SEARCH")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(f"Symbol: {symbol}  Interval: {interval}  Cross-Asset: {cross_asset}")
    if meta:
        print(f"Meta: {meta}")
    print(f"Matches: {len(matches)}")

    # show top 3
    for i, m in enumerate(matches[:3], start=1):
        sim = m.get("similarity")
        mid = m.get("id") or m.get("matchId")
        dt = m.get("date") or m.get("ts")
        msym = m.get("symbol") or symbol
        print(f"#{i}: {msym:10s} id={mid} similarity={sim:.4f} date/ts={dt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
