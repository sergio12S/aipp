from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from .client import Client


class BacktestValidator:
    """
    Tools for validating backtest integrity and detecting data leakage.
    """

    def __init__(self, client: Client):
        self.client = client

    def check_lookahead_leakage(
        self, symbol: str, interval: str, timestamp: int, q: int = 40, f: int = 24
    ) -> Dict[str, Any]:
        """
        Checks if the API returns patterns from the 'future' relative to a given timestamp.
        In a strict backtest, no pattern should have a timestamp > (current_timestamp - q - f).
        """
        # We use the RL simple endpoint as it's a good way to get many matches for a point in time
        try:
            resp = self.client.get_rl_simple(
                symbol=symbol,
                interval=interval,
                timestamp=timestamp,
                end_ts=timestamp,  # CRITICAL: Tell the engine to only look at the past
                forecast_horizon=f,
                num_episodes=100,
                min_similarity=0.5,
            )
            episodes = resp.get("episodes", [])
        except Exception as e:
            return {
                "error": str(e),
                "passed": False,
                "total_checked": 0,
                "future_matches_found": 0,
                "details": [],
            }

        future_eps = [ep for ep in episodes if ep["timestamp"] > timestamp]

        passed = len(future_eps) == 0

        return {
            "passed": passed,
            "total_checked": len(episodes),
            "future_matches_found": len(future_eps),
            "anchor_date": datetime.fromtimestamp(
                timestamp / 1000, tz=timezone.utc
            ).isoformat(),
            "details": [
                {
                    "ts": ep["timestamp"],
                    "date": datetime.fromtimestamp(
                        ep["timestamp"] / 1000, tz=timezone.utc
                    ).isoformat(),
                }
                for ep in future_eps[:5]
            ],
        }

    def audit_strategy_consistency(
        self,
        symbol: str,
        interval: str,
        periods: List[Dict[str, Any]],
        **backtest_params,
    ) -> Dict[str, Any]:
        """
        Runs backtests across multiple periods and checks for unrealistic consistency.
        If win rate is exactly the same across bear/bull markets, it might indicate an issue.
        """
        results = []
        for period in periods:
            start_ts = int(period["start"].timestamp() * 1000)
            end_ts = int(period["end"].timestamp() * 1000)

            res = self.client.backtest(
                symbol=symbol,
                interval=interval,
                start_ts=start_ts,
                end_ts=end_ts,
                include_stats=True,
                **backtest_params,
            )

            stats = res.get("stats", {})
            results.append(
                {
                    "name": period["name"],
                    "win_rate": stats.get("winRate", 0),
                    "profit_factor": stats.get("profitFactor", 0),
                    "total_return": stats.get("totalReturnPct", 0),
                }
            )

        win_rates = [r["win_rate"] for r in results]
        variance = max(win_rates) - min(win_rates) if win_rates else 0

        # Heuristic: Real strategies vary between 5% and 30% across different regimes
        is_realistic = 3.0 <= variance <= 40.0

        return {
            "is_realistic": is_realistic,
            "win_rate_variance": variance,
            "period_results": results,
        }
