from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aipricepatterns import BacktestAuditor, Client


def _period_start_ms(days: int) -> int:
    return int((time.time() - days * 24 * 60 * 60) * 1000)


def _read_watchlist(path: Path) -> List[str]:
    symbols: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        symbols.append(s)
    return symbols


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _grid_hint_from_regime(regime_label: Optional[str]) -> str:
    if not regime_label:
        return ""

    r = regime_label.upper()
    if "MEAN_REVERSION" in r or "RANGE" in r:
        return "Grid-friendly"
    if "MOMENTUM" in r or "BREAKOUT" in r:
        return "Trend-only"
    if "VOLATILITY" in r:
        return "High-vol (widen steps)"
    if "DOWNTREND" in r or "UPTREND" in r:
        return "Trending (bias-aware)"
    return "Mixed"


def _get_stats(backtest_res: Dict[str, Any]) -> Dict[str, Any]:
    s = backtest_res.get("stats")
    return s if isinstance(s, dict) else {}


def _fmt_pct(v: Any, *, digits: int = 2) -> str:
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:+.{digits}f}%"


def _fmt_pp_delta(a: Any, b: Any, *, digits: int = 2) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "n/a"
    return f"{(a - b):+.{digits}f} pp"


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


def _scan_watchlist(
    client: Client,
    *,
    symbols: List[str],
    interval: str,
    q: int,
    f: int,
    limit: int,
    blocked_regimes: List[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    blocked = set(blocked_regimes)

    reqs = [
        {
            "symbol": s,
            "interval": interval,
            "q": q,
            "f": f,
            "limit": limit,
            "sort": "similarity",
        }
        for s in symbols
    ]
    batch = client.batch_search(reqs)
    results = batch.get("results") if isinstance(batch, dict) else None
    if not isinstance(results, list) or len(results) != len(symbols):
        results = [{} for _ in symbols]

    rows: List[Dict[str, Any]] = []

    for i, sym in enumerate(symbols):
        search_res = (
            results[i] if i < len(results) and isinstance(results[i], dict) else {}
        )
        matches = search_res.get("matches") or []
        match_count = len(matches) if isinstance(matches, list) else 0

        regime_id = None
        regime_error = None
        try:
            r = client.get_current_regime(sym, interval)
            regime_id = _safe_get(r, ["currentRegime", "id"], None)
        except Exception as e:
            regime_error = str(e)

        decision = "NO"
        reason = ""
        if regime_error:
            reason = f"regime_error:{regime_error}"
        elif regime_id and regime_id in blocked:
            reason = f"blocked_regime:{regime_id}"
        elif match_count == 0:
            reason = "no_matches"
        else:
            decision = "GO"
            reason = "ok"

        rows.append(
            {
                "symbol": sym,
                "regime": regime_id,
                "matches": match_count,
                "decision": decision,
                "reason": reason,
            }
        )

    # choose best GO candidate
    go = [r for r in rows if r.get("decision") == "GO"]
    go_sorted = sorted(go, key=lambda r: int(r.get("matches") or 0), reverse=True)
    pick = go_sorted[0].get("symbol") if go_sorted else None

    return rows, pick


def main() -> int:
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    interval = os.getenv("AIPP_DEMO_INTERVAL", "1h")
    q = int(os.getenv("AIPP_DEMO_Q", "60"))
    f = int(os.getenv("AIPP_DEMO_F", "24"))
    limit = int(os.getenv("AIPP_DEMO_LIMIT", "16"))

    days = int(os.getenv("AIPP_DEMO_DAYS", "90"))
    step = int(os.getenv("AIPP_DEMO_STEP", "24"))
    min_prob = float(os.getenv("AIPP_DEMO_MIN_PROB", "0.50"))

    fee_pct = _env_float("AIPP_FEE_PCT", "AIPP_DEMO_FEE_PCT", default=0.04)
    slippage_pct = _env_float(
        "AIPP_SLIPPAGE_PCT", "AIPP_DEMO_SLIPPAGE_PCT", default=0.02
    )

    blocked_regimes = [
        s.strip()
        for s in os.getenv(
            "AIPP_BLOCK_REGIMES", "BEARISH_MOMENTUM,STABLE_DOWNTREND"
        ).split(",")
        if s.strip()
    ]

    client = Client(base_url=base_url, api_key=api_key)

    watchlist_path = Path(__file__).with_name("watchlist.txt")
    symbols = _read_watchlist(watchlist_path)

    print("=" * 78)
    print("INVESTOR ONE-PAGER — WATCHLIST → BACKTEST → REGIME AUDIT")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(f"Watchlist: {', '.join(symbols)}")
    print(f"Interval: {interval}  q={q}  f={f}  limit={limit}")
    print(
        f"Blocked regimes: {', '.join(blocked_regimes) if blocked_regimes else '(none)'}"
    )

    rows, pick = _scan_watchlist(
        client,
        symbols=symbols,
        interval=interval,
        q=q,
        f=f,
        limit=limit,
        blocked_regimes=blocked_regimes,
    )

    print("-")
    print("Scan results:")
    for r in rows:
        print(
            f"  {r['symbol']:10s} regime={str(r.get('regime') or 'N/A'):18s} matches={int(r.get('matches') or 0):3d}  {r['decision']:>3s}  {r['reason']}"
        )

    if not pick:
        print("-")
        print("Result: NO-GO for entire watchlist (no safe candidates right now).")
        return 0

    print("-")
    print(f"Selected candidate: {pick} (best GO by match count)")

    # Grid trading intel (regime + suggested grid params)
    try:
        grid = client.get_grid_stats(pick, interval)
        g_regime = _safe_get(grid, ["regime", "label"], None)
        g_conf = _safe_get(grid, ["regime", "confidence"], None)
        g_hint = _grid_hint_from_regime(g_regime if isinstance(g_regime, str) else None)

        bias = _safe_get(grid, ["gridRecommendation", "bias"], None)
        step_pct = _safe_get(grid, ["gridRecommendation", "suggestedStepPct"], None)
        lower_pct = _safe_get(grid, ["gridRecommendation", "lowerPct"], None)
        upper_pct = _safe_get(grid, ["gridRecommendation", "upperPct"], None)
        levels = _safe_get(grid, ["gridRecommendation", "levels"], None)

        print("-")
        print("Grid intel (current):")
        if isinstance(g_regime, str) and g_regime.strip():
            if isinstance(g_conf, (int, float)):
                print(f"  Regime: {g_regime} (conf={g_conf:.2f})")
            else:
                print(f"  Regime: {g_regime}")
        if g_hint:
            print(f"  Hint:   {g_hint}")

        parts: List[str] = []
        if isinstance(bias, str) and bias:
            parts.append(f"bias={bias}")
        if isinstance(levels, int):
            parts.append(f"levels={levels}")
        if isinstance(step_pct, (int, float)):
            parts.append(f"step≈{step_pct:.4f}%")
        if isinstance(lower_pct, (int, float)) and isinstance(upper_pct, (int, float)):
            parts.append(f"range=[{lower_pct:+.3f}%, {upper_pct:+.3f}%]")
        if parts:
            print("  Grid:   " + "  ".join(parts))
    except Exception:
        pass

    start_ts = _period_start_ms(days)

    bt_no_cost = client.backtest(
        symbol=pick,
        interval=interval,
        q=q,
        f=f,
        step=step,
        top_k=5,
        min_prob=min_prob,
        start_ts=start_ts,
        include_stats=True,
        fee_pct=0.0,
        slippage_pct=0.0,
    )

    bt_with_cost = client.backtest(
        symbol=pick,
        interval=interval,
        q=q,
        f=f,
        step=step,
        top_k=5,
        min_prob=min_prob,
        start_ts=start_ts,
        include_stats=True,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )

    s0 = _get_stats(bt_no_cost)
    s1 = _get_stats(bt_with_cost)

    print("-")
    print("Backtest reality check (walk-forward):")
    print("  No-cost:")
    print(f"    totalReturnPct: {_fmt_pct(s0.get('totalReturnPct'))}")
    print(f"    winRate:        {s0.get('winRate', 'n/a')}")
    print(f"    profitFactor:   {s0.get('profitFactor', 'n/a')}")
    print(f"    maxDrawdownPct: {s0.get('maxDrawdownPct', 'n/a')}")
    print(f"  With costs (feePct={fee_pct:.4f}%  slippagePct={slippage_pct:.4f}%):")
    print(
        f"    totalReturnPct: {_fmt_pct(s1.get('totalReturnPct'))}  (Δ {_fmt_pp_delta(s1.get('totalReturnPct'), s0.get('totalReturnPct'))})"
    )
    print(f"    winRate:        {s1.get('winRate', 'n/a')}")
    print(f"    profitFactor:   {s1.get('profitFactor', 'n/a')}")
    print(f"    maxDrawdownPct: {s1.get('maxDrawdownPct', 'n/a')}")

    trades = bt_with_cost.get("results") or bt_with_cost.get("trades") or []
    trades = trades if isinstance(trades, list) else []

    returns = [t.get("actualReturnPct", 0.0) for t in trades if isinstance(t, dict)]
    wins = [r for r in returns if isinstance(r, (int, float)) and r > 0]
    losses = [r for r in returns if isinstance(r, (int, float)) and r < 0]

    print("-")
    print("Backtest quick stats:")
    print(f"  Period: last {days} days")
    print(f"  Signals: {len(trades)}  Wins/Losses: {len(wins)}/{len(losses)}")
    if returns:
        total = sum(r for r in returns if isinstance(r, (int, float)))
        print(f"  Total return (sum of trade returns): {total:+.2f}%")

    trades_for_audit: List[Dict[str, Any]] = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = t.get("ts")
        if not isinstance(ts, int):
            continue
        trades_for_audit.append(
            {"entryTime": ts, "returnPct": t.get("actualReturnPct", 0.0)}
        )

    auditor = BacktestAuditor(client)
    report = auditor.analyze_losses(pick, interval, trades_for_audit)
    total_losses = report.get("total_losses", 0)
    dist = report.get("regime_distribution", {}) or {}

    print("-")
    print("Regime audit (losses):")
    print(f"  Total losses analyzed: {total_losses}")
    if isinstance(dist, dict) and dist:
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        for rid, cnt in items[:6]:
            pct = (cnt / total_losses * 100) if total_losses else 0
            print(f"  {rid:20s}: {cnt:3d} ({pct:4.1f}%)")
        avoid = items[0][0]
        print(f"  Recommendation: avoid trading in regime = {avoid}")

    print("-")
    print("Result: GO (candidate selected + audited).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
