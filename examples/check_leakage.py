#!/usr/bin/env python3
"""Check for data leakage in RL training"""

import os
from datetime import datetime, timezone
from aipricepatterns import Client, BacktestValidator


def ts_to_date(ts):
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def date_to_ts(date_str, time_str="00:00"):
    """Convert human-readable date to timestamp
    Args:
        date_str: Date in format "YYYY-MM-DD"
        time_str: Time in format "HH:MM" (default "00:00")
    Returns:
        Timestamp in milliseconds
    """
    dt_str = f"{date_str} {time_str}"
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


CURRENT_TS = date_to_ts("2024-01-01", "00:00")


def main():
    # Default to the Rust API proxy path on production
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)
    validator = BacktestValidator(client)

    print("=== DATA LEAKAGE AUDIT ===")
    print(f"Target Server: {base_url}")

    # Check dataset status first
    audit_ts = CURRENT_TS
    try:
        status = client.get_dataset_status()
        print(f"Server Status: Connected")
        if "datasets" in status:
            for ds in status["datasets"]:
                if ds["symbol"] == "BTCUSDT" and ds["interval"] == "1h":
                    print(f"Dataset:       BTCUSDT 1h ({ds['count']} candles)")
                    # Pick a point 2000 candles before the end to make the test "honest"
                    if "end_ts" in ds and ds["end_ts"]:
                        audit_ts = ds["end_ts"] - (2000 * 3600 * 1000)
                        print(
                            f"Audit Point:   {ts_to_date(audit_ts)} (Selected inside history)"
                        )
    except Exception:
        print(f"Server Status: Warning (Could not fetch dataset status)")

    # Use the new validator logic
    try:
        res = validator.check_lookahead_leakage(
            symbol="BTCUSDT", interval="1h", timestamp=audit_ts, q=40, f=24
        )
    except Exception as e:
        print(f"\n⚠️ CONNECTION ERROR: {e}")
        print("Check if AIPP_BASE_URL is correct and the server is reachable.")
        return

    if res.get("error"):
        print(f"\n⚠️ API ERROR: {res['error']}")
        if "404" in res["error"]:
            print(
                "Hint: 404 usually means the base_url is wrong (e.g. missing /api/rust suffix)."
            )
        return

    if res["total_checked"] == 0:
        print("\n⚠️ WARNING: No historical matches found for this timestamp.")
        print(
            "This could mean the timestamp is out of range or similarity threshold is too high."
        )
        return

    if res["passed"]:
        print(f"\n✅ PASS: No patterns from the future were returned.")
        print(f"Checked {res['total_checked']} historical matches.")
    else:
        print(
            f"\n❌ FAIL: Found {res['future_matches_found']} patterns from the future!"
        )
        for detail in res["details"]:
            print(f"  - Match at {detail['date']} (is after query point)")

    print("\n=== EXPLANATION ===")
    print("""
    The engine must strictly respect time causality. 
    When searching for patterns at time T, it must only return matches 
    that were fully completed (including their forecast horizon) before T.
    """)


if __name__ == "__main__":
    main()
