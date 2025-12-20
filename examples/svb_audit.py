import os
import datetime
from aipricepatterns import Client

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1h"

# Target: March 11, 2023 (Peak SVB Panic / USDC Depeg)
target_date = datetime.datetime(2023, 3, 11, 12, 0, 0, tzinfo=datetime.timezone.utc)
target_ts = int(target_date.timestamp() * 1000)


def run_audit():
    print("--- AUDIT: SVB Crisis (USDC Depeg) ---")
    print(f"Target Date: {target_date}")
    print(f"Timestamp: {target_ts}")

    # For production use: https://aipricepatterns.com/api/rust
    # For local use: http://localhost:8787
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    try:
        # Use the SDK to get pattern metrics
        data = client.get_pattern_metrics(
            symbol=SYMBOL,
            interval=INTERVAL,
            q=60,
            f=48,
            limit=100,
            window_end_ts=target_ts,
        )

        meta = data.get("meta", {})
        metrics = data.get("metrics", {})

        print("\n[1] Context")
        print(f"Price at moment: ${meta.get('lastPrice', 'N/A')}")

        print("\n[2] Predictive Analytics")
        print(f"Similar Patterns Found: {metrics.get('count', 0)}")
        print(f"Win Rate (Up Prob): {metrics.get('upProbPct', 0):.2f}%")
        print(f"Average Return: {metrics.get('averagePct', 0):.2f}%")
        print(f"Best Case: {metrics.get('bestPct', 0):.2f}%")
        print(f"Worst Case: {metrics.get('worstPct', 0):.2f}%")

        print("\n[3] Risk Profile")
        print(f"Projected Volatility: {metrics.get('avgVolatilityPct', 0):.2f}%")
        print(f"Median Drawdown: {metrics.get('medianDrawdownPct', 0):.2f}%")

        # 2. Interpretation
        win_rate = metrics.get("upProbPct", 0)
        if win_rate > 65:
            print("\n>>> CONCLUSION: STRONG BUY (Contrarian Signal)")
            print(
                "While the market was panicking, the historical patterns indicated a high probability of a bounce."
            )
        elif win_rate < 35:
            print("\n>>> CONCLUSION: STRONG SELL (Trend Continuation)")
        else:
            print("\n>>> CONCLUSION: UNCERTAIN / HOLD")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_audit()
