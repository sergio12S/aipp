#!/usr/bin/env python3
"""
Institutional Evidence Generator
===============================
Automates the collection of evidence for institutional due diligence.
- Look-ahead bias (Leakage) verification.
- Multi-regime performance analysis.
- Statistical significance (Alpha) proof.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# --- SDK Import Helper ---
repo_src = (Path(__file__).parent.parent.parent / "src").resolve()
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

try:
    from aipricepatterns import BacktestValidator, Client
except ImportError:
    print("Error: aipricepatterns SDK not found.")
    sys.exit(1)

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")


def main():
    client = Client(base_url=BASE_URL)
    validator = BacktestValidator(client)

    evidence = {
        "timestamp": datetime.now().isoformat(),
        "config": {"symbol": SYMBOL, "interval": INTERVAL},
        "leakage_test": None,
        "regime_performance": [],
    }

    print(f"--- Generating Evidence Base for {SYMBOL} ({INTERVAL}) ---")

    # 1. Leakage Test (Proof of Integrity)
    print("Step 1: Running Data Leakage Audit...")
    # Check a point 6 months ago
    six_months_ago = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
    leakage_res = validator.check_lookahead_leakage(SYMBOL, INTERVAL, six_months_ago)
    evidence["leakage_test"] = leakage_res
    print(f"Leakage Test: {'PASSED' if leakage_res['passed'] else 'FAILED'}")

    # 2. Multi-Regime Performance
    print("\nStep 2: Testing Strategy across Market Regimes...")
    regimes = [
        {"name": "Bear/Neutral 2022", "start": "2022-01-01", "end": "2022-12-31"},
        {"name": "Bull Market 2024", "start": "2024-01-01", "end": "2024-11-20"},
    ]

    for reg in regimes:
        print(f"  Testing {reg['name']}...")
        start_ts = int(datetime.strptime(reg["start"], "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(reg["end"], "%Y-%m-%d").timestamp() * 1000)

        try:
            res = client.backtest(
                symbol=SYMBOL,
                interval=INTERVAL,
                start_ts=start_ts,
                end_ts=end_ts,
                step=24,
                include_stats=True,
            )
            stats = res.get("stats", {})
            evidence["regime_performance"].append(
                {
                    "period": reg["name"],
                    "total_return": stats.get("totalReturnPct", 0),
                    "benchmark": stats.get("benchmarkReturnPct", 0),
                    "profit_factor": stats.get("profitFactor", 0),
                    "win_rate": stats.get("winRate", 0),
                    "sharpe": stats.get("sharpeRatio", 0),
                }
            )
        except Exception as e:
            print(f"  Failed to test {reg['name']}: {e}")

    # 3. Save Results
    output_path = Path(__file__).parent / "evidence_data.json"
    with open(output_path, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"\nEvidence data saved to {output_path}")


if __name__ == "__main__":
    main()
