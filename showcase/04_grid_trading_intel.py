from __future__ import annotations

import os
from typing import Any, Dict, List

from aipricepatterns import Client


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _strategy_hint(regime_label: str | None) -> str:
    if not regime_label:
        return "No regime label (fallback to conservative sizing)."

    r = regime_label.upper()
    if "MEAN_REVERSION" in r:
        return "Mean reversion → Grid bots make sense (tight steps, many levels)."
    if "MOMENTUM" in r or "BREAKOUT" in r:
        return (
            "Momentum/breakout → Grid is riskier; prefer trend-following / wider grids."
        )
    if "VOLATILITY" in r:
        return "High volatility → widen grid steps, reduce leverage/levels."
    if "RANGE" in r:
        return "Range → Grid-friendly; focus on mean/σ bands."
    if "DOWNTREND" in r or "UPTREND" in r:
        return "Trending → consider trend mode or asymmetric grid (bias-aware)."

    return "Use regime + volatility to choose grid width/step."


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    symbols = [
        s.strip()
        for s in os.getenv("AIPP_GRID_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
        if s.strip()
    ]
    interval = os.getenv("AIPP_GRID_INTERVAL", "1h")

    client = Client(base_url=base_url, api_key=api_key)

    print("=" * 78)
    print("GRID TRADING INTEL — REGIME + VOLATILITY + σ LEVELS")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(f"Symbols:  {', '.join(symbols)}")
    print(f"Interval: {interval}")

    for sym in symbols:
        r = client.get_grid_stats(sym, interval)

        regime_label = _safe_get(r, ["regime", "label"], None)
        regime_conf = _safe_get(r, ["regime", "confidence"], None)
        up_prob = _safe_get(r, ["summary", "upProbPct"], None)
        vol = _safe_get(r, ["summary", "riskAnalysis", "volatilityForecast"], None)

        bias = _safe_get(r, ["gridRecommendation", "bias"], None)
        step = _safe_get(r, ["gridRecommendation", "suggestedStepPct"], None)
        lower = _safe_get(r, ["gridRecommendation", "lowerPct"], None)
        upper = _safe_get(r, ["gridRecommendation", "upperPct"], None)
        levels = _safe_get(r, ["gridRecommendation", "levels"], None)
        zone = _safe_get(r, ["priceContext", "zone"], None)

        print("-")
        print(f"{sym} ({interval})")

        if regime_label is not None:
            if isinstance(regime_conf, (int, float)):
                print(f"  Regime:  {regime_label} (conf={regime_conf:.2f})")
            else:
                print(f"  Regime:  {regime_label}")

        if isinstance(up_prob, (int, float)):
            print(f"  UpProb:  {up_prob:.1f}%")
        if isinstance(vol, (int, float)):
            print(f"  Vol:     {vol:.4f}%")

        parts: List[str] = []
        if bias:
            parts.append(f"bias={bias}")
        if isinstance(levels, int):
            parts.append(f"levels={levels}")
        if isinstance(step, (int, float)):
            parts.append(f"step≈{step:.4f}%")
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            parts.append(f"range=[{lower:+.3f}%, {upper:+.3f}%]")
        if zone:
            parts.append(f"zone={zone}")

        if parts:
            print("  Grid:    " + "  ".join(parts))

        print("  Hint:    " + _strategy_hint(regime_label))

    print("-")
    print("Tip: run `aipp grid --raw` for full JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
