from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from aipricepatterns import Client


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _period_anchor_ms(days_ago: int) -> int:
    return int((time.time() - days_ago * 24 * 60 * 60) * 1000)


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    symbol = os.getenv("AIPP_RL_SYMBOL", "BTCUSDT")
    interval = os.getenv("AIPP_RL_INTERVAL", "1h")

    # Recommended: use anchorTs for a “known” moment (in ms). If not set, use recent anchor.
    anchor_ts = int(os.getenv("AIPP_RL_ANCHOR_TS", str(_period_anchor_ms(30))))

    forecast_horizon = int(os.getenv("AIPP_RL_HORIZON", "24"))
    num_episodes = int(os.getenv("AIPP_RL_NUM_EPISODES", "50"))
    min_similarity = float(os.getenv("AIPP_RL_MIN_SIMILARITY", "0.80"))

    batch_query_length = int(os.getenv("AIPP_RL_QUERY_LENGTH", "40"))
    batch_size = int(os.getenv("AIPP_RL_BATCH_SIZE", "500"))

    client = Client(base_url=base_url, api_key=api_key)

    print("=" * 78)
    print("RL — PARALLEL UNIVERSES (EPISODES) + TENSOR FACTORY (TRAINING-BATCH)")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(
        f"Symbol: {symbol}  Interval: {interval}  anchorTs={anchor_ts}  horizon={forecast_horizon}"
    )

    episodes_res = client.get_rl_episodes(
        symbol=symbol,
        interval=interval,
        anchor_ts=anchor_ts,
        forecast_horizon=forecast_horizon,
        num_episodes=num_episodes,
        min_similarity=min_similarity,
        include_actions=False,
        reward_type="returns",
        sampling_strategy="uniform",
    )

    meta = _safe_get(episodes_res, ["meta"], {})
    episodes = _safe_get(episodes_res, ["episodes"], [])
    statistics = _safe_get(episodes_res, ["statistics"], {})
    risk = _safe_get(episodes_res, ["riskAnalysis"], {})

    print("-")
    print("Episodes:")
    print(f"  Returned: {len(episodes) if isinstance(episodes, list) else 'n/a'}")

    regime_type = meta.get("regimeType") if isinstance(meta, dict) else None
    regime_conf = meta.get("regimeConfidence") if isinstance(meta, dict) else None
    if regime_type is not None:
        if isinstance(regime_conf, (int, float)):
            print(f"  Regime:  {regime_type} (conf={regime_conf:.2f})")
        else:
            print(f"  Regime:  {regime_type}")

    if isinstance(episodes, list) and episodes:
        sims = [
            e.get("similarity")
            for e in episodes
            if isinstance(e, dict) and isinstance(e.get("similarity"), (int, float))
        ]
        if sims:
            print(
                f"  Similarity: min={min(sims):.3f}  mean={sum(sims) / len(sims):.3f}  max={max(sims):.3f}"
            )

    if isinstance(statistics, dict) and statistics:
        print("  Stats:")
        for k in ("avgReturn", "winRate", "avgMaxDrawdown"):
            if k in statistics:
                print(f"    {k}: {statistics.get(k)}")

    if isinstance(risk, dict) and risk:
        print("  Risk:")
        for k in (
            "volatilityForecast",
            "valueAtRisk95",
            "downsideProbability",
            "crashProbability",
        ):
            if k in risk:
                print(f"    {k}: {risk.get(k)}")

    print("-")
    print("Training batch (offline RL):")
    batch_res = client.get_rl_training_batch(
        symbol=symbol,
        interval=interval,
        query_length=batch_query_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        min_similarity=0.70,
    )

    data = _safe_get(batch_res, ["data"], {})
    if isinstance(data, dict) and data:
        states = data.get("states")
        next_states = data.get("nextStates")
        rewards = data.get("rewards")
        dones = data.get("dones")
        print(f"  states:     {len(states) if isinstance(states, list) else 'n/a'}")
        print(
            f"  nextStates: {len(next_states) if isinstance(next_states, list) else 'n/a'}"
        )
        print(f"  rewards:    {len(rewards) if isinstance(rewards, list) else 'n/a'}")
        print(f"  dones:      {len(dones) if isinstance(dones, list) else 'n/a'}")

    print("-")
    print("Takeaway:")
    print(
        "  Traders: train a specialist policy per regime/context instead of one global agent."
    )
    print(
        "  Investors: the moat is data retrieval — we generate 'parallel universes' on demand."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
