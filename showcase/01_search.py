from __future__ import annotations

import os

from aipricepatterns import Client


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    client = Client(base_url=base_url, api_key=api_key)

    symbol = os.getenv("AIPP_DEMO_SYMBOL", "BTCUSDT")
    interval = os.getenv("AIPP_DEMO_INTERVAL", "1h")

    res = client.search(symbol=symbol, interval=interval, q=60, f=24, top_k=10)

    matches = res.get("matches") or []
    meta = res.get("meta") or {}

    print("=" * 78)
    print("DEMO 01 — SEARCH")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(f"Symbol: {symbol}  Interval: {interval}")
    if meta:
        print(f"Meta: {meta}")
    print(f"Matches: {len(matches)}")

    # show top 3
    for i, m in enumerate(matches[:3], start=1):
        sim = m.get("similarity")
        mid = m.get("id") or m.get("matchId")
        dt = m.get("date") or m.get("ts")
        print(f"#{i}: id={mid} similarity={sim} date/ts={dt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
