from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from aipricepatterns import Client


def _period_start_ms(days: int) -> int:
    return int((time.time() - days * 24 * 60 * 60) * 1000)


def _get_stats(backtest_res: Dict[str, Any]) -> Dict[str, Any]:
    s = backtest_res.get("stats")
    return s if isinstance(s, dict) else {}


def _fmt_pct(v: Optional[float]) -> str:
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:+.2f}%"


def _env_float(*names: str, default: float) -> float:
    for n in names:
        v = os.getenv(n)
        if v is None or v.strip() == "":
            continue
        try:
            return float(v)
        except ValueError:
            continue
    return default


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    symbol = os.getenv("AIPP_BT_SYMBOL", "BTCUSDT")
    interval = os.getenv("AIPP_BT_INTERVAL", "1h")
    q = int(os.getenv("AIPP_BT_Q", "24"))
    f = int(os.getenv("AIPP_BT_F", "12"))
    step = int(os.getenv("AIPP_BT_STEP", "24"))
    top_k = int(os.getenv("AIPP_BT_TOPK", "5"))
    min_prob = float(os.getenv("AIPP_BT_MIN_PROB", "0.50"))
    days = int(os.getenv("AIPP_BT_DAYS", "90"))

    fee_pct = _env_float("AIPP_FEE_PCT", "AIPP_BT_FEE_PCT", default=0.04)
    slippage_pct = _env_float("AIPP_SLIPPAGE_PCT", "AIPP_BT_SLIPPAGE_PCT", default=0.02)

    client = Client(base_url=base_url, api_key=api_key)

    start_ts = _period_start_ms(days)

    print("=" * 78)
    print("BACKTESTING — REALITY CHECK (WITH COSTS)")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(
        f"Symbol: {symbol}  Interval: {interval}  q={q}  f={f}  step={step}  topK={top_k}  minProb={min_prob}"
    )
    print(f"Period: last {days} days")

    bt0 = client.backtest(
        symbol=symbol,
        interval=interval,
        q=q,
        f=f,
        step=step,
        top_k=top_k,
        min_prob=min_prob,
        start_ts=start_ts,
        include_stats=True,
        fee_pct=0.0,
        slippage_pct=0.0,
    )

    bt1 = client.backtest(
        symbol=symbol,
        interval=interval,
        q=q,
        f=f,
        step=step,
        top_k=top_k,
        min_prob=min_prob,
        start_ts=start_ts,
        include_stats=True,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )

    s0 = _get_stats(bt0)
    s1 = _get_stats(bt1)

    print("-")
    print("Report card (no costs):")
    print(f"  totalReturnPct: {_fmt_pct(s0.get('totalReturnPct'))}")
    print(f"  winRate:        {s0.get('winRate', 'n/a')}")
    print(f"  profitFactor:   {s0.get('profitFactor', 'n/a')}")
    print(f"  maxDrawdownPct: {s0.get('maxDrawdownPct', 'n/a')}")

    print("-")
    print(f"Report card (feePct={fee_pct:.4f}%  slippagePct={slippage_pct:.4f}%):")
    print(f"  totalReturnPct: {_fmt_pct(s1.get('totalReturnPct'))}")
    print(f"  winRate:        {s1.get('winRate', 'n/a')}")
    print(f"  profitFactor:   {s1.get('profitFactor', 'n/a')}")
    print(f"  maxDrawdownPct: {s1.get('maxDrawdownPct', 'n/a')}")

    print("-")
    print("Investor takeaway:")
    print("  This backtest is walk-forward (no look-ahead) and includes friction.")
    print("  If performance collapses under realistic costs, it’s not deployable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
