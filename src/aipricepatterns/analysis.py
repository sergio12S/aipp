from typing import Any, Dict, List

import numpy as np

from .client import Client


class SignalStabilityAnalyzer:
    """
    Analyzes the stability of a pattern signal by checking its persistence
    across small time shifts.
    """

    def __init__(self, client: Client):
        self.client = client

    def check_stability(
        self,
        symbol: str,
        interval: str,
        base_offset: int,
        offsets: List[int] = [0, 1, 2],
        **search_params,
    ) -> Dict[str, Any]:
        """
        Checks if the forecast is stable across small time shifts.

        :param symbol: The trading pair symbol.
        :param interval: Timeframe (e.g., '1h').
        :param base_offset: The historical offset (candles from end) to check.
        :param offsets: List of additional offsets to check relative to base.
                        e.g. [0, 1, 2] checks base, base+1, base+2.
        :param search_params: Additional params for recalc_patterns (q, f, etc.)
        """
        results = []
        scores = []

        for drift in offsets:
            # Increasing offset moves further back in time
            check_offset = base_offset + drift

            try:
                res = self.client.recalc_patterns(
                    symbol=symbol,
                    interval=interval,
                    start=check_offset,
                    **search_params,
                )

                if (
                    "forecast" in res
                    and res["forecast"]
                    and res["forecast"].get("median")
                ):
                    median = res["forecast"]["median"]
                    # Calculate return from the start of the forecast
                    start_price = median[0]
                    end_price = median[-1]
                    pct = (end_price - start_price) / start_price * 100
                    scores.append(pct)
                    results.append(
                        {"offset": drift, "forecast_pct": pct, "found": True}
                    )
                else:
                    results.append({"offset": drift, "found": False})
            except Exception as e:
                results.append({"offset": drift, "error": str(e)})

        if not scores:
            return {"stable": False, "reason": "No forecasts found", "details": results}

        std_dev = np.std(scores)
        mean_forecast = np.mean(scores)

        # Heuristic: If std_dev is low relative to the mean (or absolute), it's stable.
        # A threshold of 0.5% volatility in the forecast is a reasonable starting point.
        is_stable = std_dev < 0.5

        return {
            "stable": is_stable,
            "volatility": float(std_dev),
            "mean_forecast": float(mean_forecast),
            "min_forecast": float(np.min(scores)),
            "max_forecast": float(np.max(scores)),
            "details": results,
        }


class BacktestAuditor:
    """
    Audits backtest results to find systematic failures, such as specific
    market regimes where the strategy underperforms.
    """

    def __init__(self, client: Client):
        self.client = client

    def analyze_losses(
        self, symbol: str, interval: str, trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes losing trades to find common regimes.

        :param trades: List of trade objects from the backtest result.
        """
        losing_trades = [t for t in trades if t.get("actualReturnPct", 0) < 0]
        regime_counts = {}

        print(f"Analyzing {len(losing_trades)} losing trades...")

        for trade in losing_trades:
            ts = trade.get("ts")
            if not ts:
                continue

            try:
                r_info = self.client.detect_regime(symbol, interval, timestamp=ts)
                if "currentRegime" in r_info and r_info["currentRegime"]:
                    r_id = r_info["currentRegime"]["id"]
                    regime_counts[r_id] = regime_counts.get(r_id, 0) + 1
            except Exception as e:
                print(f"Error detecting regime for {ts}: {e}")

        return {
            "total_losses": len(losing_trades),
            "regime_distribution": regime_counts,
        }
