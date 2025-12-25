"""aipricepatterns.cli

Console entrypoint for the AI Price Patterns Python SDK.

Installs a `aipp` command via pyproject.toml [project.scripts].

This module intentionally keeps CLI / printing logic out of the SDK core.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .analysis import BacktestAuditor
from .client import Client

# For production use: https://aipricepatterns.com/api/rust
# For local use: http://localhost:8787
DEFAULT_BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
DEFAULT_API_KEY = os.getenv("AIPP_API_KEY")


def _env_float(*names: str) -> Optional[float]:
    for name in names:
        v = os.getenv(name)
        if v is None or v.strip() == "":
            continue
        try:
            return float(v)
        except ValueError:
            continue
    return None


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def _fmt_dt(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_float_list(value: str) -> List[float]:
    s = (value or "").strip()
    if not s:
        return []

    # JSON array
    if s.startswith("["):
        data = json.loads(s)
        if not isinstance(data, list):
            raise ValueError("currentState must be a JSON array")
        out: List[float] = []
        for x in data:
            if not isinstance(x, (int, float)):
                raise ValueError("currentState must contain only numbers")
            out.append(float(x))
        return out

    # Comma-separated
    out2: List[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out2.append(float(p))
    return out2


@dataclass
class CommonArgs:
    base_url: str
    api_key: Optional[str]


def _make_client(common: CommonArgs) -> Client:
    return Client(api_key=common.api_key, base_url=common.base_url)


def cmd_search(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.search(
        symbol=args.symbol,
        interval=args.interval,
        q=args.q,
        f=args.f,
        limit=args.limit,
        sort=args.sort,
        force=args.force,
        cross_asset=getattr(args, "cross_asset", False),
    )

    matches = res.get("matches") or []
    meta = res.get("meta") or {}
    forecast = res.get("forecast") or {}

    print("=" * 78)
    print("PATTERN SEARCH")
    print("=" * 78)
    print(f"Base URL: {common.base_url}")
    print(
        f"Symbol:   {args.symbol}  Interval: {args.interval}  q={args.q}  f={args.f}  limit={args.limit}"
    )
    if meta:
        print(f"Meta:     {meta}")

    print(f"Matches:  {len(matches)}")
    if matches:
        print("Top matches (best effort):")
        for i, m in enumerate(matches[: min(5, len(matches))], start=1):
            sim = m.get("similarity")
            sim_pct = f"{sim * 100:.2f}%" if isinstance(sim, (int, float)) else "n/a"
            mid = m.get("id") or m.get("matchId") or "n/a"
            print(f"  #{i:02d}  similarity={sim_pct:>8}  id={mid}")

    median = forecast.get("median") if isinstance(forecast, dict) else None
    if isinstance(median, list) and len(median) >= 2:
        start_price = median[0]
        end_price = median[-1]
        if (
            isinstance(start_price, (int, float))
            and isinstance(end_price, (int, float))
            and start_price != 0
        ):
            pct = (end_price - start_price) / start_price * 100
            print(f"Forecast (median): {pct:+.2f}% over {len(median)} points")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_signals(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_signals()

    signals = res.get("signals") or []

    print("=" * 78)
    print("LIVE SIGNALS")
    print("=" * 78)
    print(f"Signals found: {len(signals)}")

    if signals:
        print(f"{'SYMBOL':10s} {'INT':4s} {'DIR':5s} {'PROB':6s} {'SIM':6s} {'TS'}")
        print("-" * 78)
        for s in signals:
            sym = s.get("symbol", "n/a")
            interval = s.get("interval", "n/a")
            direction = s.get("direction", "n/a")
            prob = s.get("up_prob")
            prob_s = f"{prob:.2f}" if isinstance(prob, (int, float)) else "n/a"
            sim = s.get("similarity")
            sim_s = f"{sim:.2f}" if isinstance(sim, (int, float)) else "n/a"
            ts = s.get("ts")
            dt = _fmt_dt(ts) if isinstance(ts, int) else "n/a"
            print(f"{sym:10s} {interval:4s} {direction:5s} {prob_s:6s} {sim_s:6s} {dt}")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_metrics(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_pattern_metrics(
        symbol=args.symbol, interval=args.interval, q=args.q, f=args.f
    )

    metrics = res.get("metrics") or {}
    ra = _safe_get(metrics, ["riskAnalysis"], {})

    print("=" * 78)
    print("PATTERN METRICS")
    print("=" * 78)
    print(f"Symbol: {args.symbol}  Interval: {args.interval}")

    if metrics:
        print(f"Count:          {metrics.get('count')}")
        print(f"AveragePct:     {metrics.get('averagePct')}")
        print(f"UpProbPct:      {metrics.get('upProbPct')}")
        if ra:
            print("Risk Analysis:")
            print(f"  volatilityForecast:   {ra.get('volatilityForecast')}")
            print(f"  valueAtRisk95:        {ra.get('valueAtRisk95')}")
            print(f"  downsideProbability:  {ra.get('downsideProbability')}")
            print(f"  crashProbability:     {ra.get('crashProbability')}")
            print(f"  kellyFraction:        {ra.get('kellyFraction')}")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_grid(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_grid_stats(symbol=args.symbol, interval=args.interval)

    summary = res.get("summary") or {}
    rec = res.get("gridRecommendation") or {}
    regime = res.get("regime") or {}
    conf = res.get("confidence") or {}
    price_ctx = res.get("priceContext") or {}
    grid_levels = res.get("gridLevels") or []
    sigma_levels = res.get("sigmaLevels") or []
    terminal = res.get("terminalPercentiles") or {}

    print("=" * 78)
    print("GRID TRADING STATS")
    print("=" * 78)
    print(f"Symbol: {args.symbol}  Interval: {args.interval}")

    if isinstance(regime, dict) and regime:
        r_label = regime.get("label")
        r_conf = regime.get("confidence")
        if r_label is not None:
            if isinstance(r_conf, (int, float)):
                print(f"Regime: {r_label}  confidence={r_conf:.2f}")
            else:
                print(f"Regime: {r_label}")

    if isinstance(conf, dict) and conf:
        c_label = conf.get("label")
        c_score = conf.get("score")
        if isinstance(c_score, (int, float)):
            if c_label:
                print(f"Model confidence: {c_label} ({c_score:.3f})")
            else:
                print(f"Model confidence: {c_score:.3f}")
        elif c_label:
            print(f"Model confidence: {c_label}")

    if summary:
        ra = _safe_get(summary, ["riskAnalysis"], {})
        print(f"Matches used: {summary.get('count')}")
        up = summary.get("upProbPct")
        stdev = summary.get("stdevPct")
        if isinstance(up, (int, float)):
            print(f"Up probability: {up:.1f}%")
        if isinstance(stdev, (int, float)):
            print(f"Volatility (stdev): {stdev:.4f}%")
        if ra:
            vf = ra.get("volatilityForecast")
            var95 = ra.get("valueAtRisk95")
            ddp = ra.get("downsideProbability")
            crash = ra.get("crashProbability")
            if isinstance(vf, (int, float)):
                print(f"Volatility forecast: {vf:.4f}%")
            if isinstance(var95, (int, float)):
                print(f"VaR95:              {var95:.4f}%")
            if isinstance(ddp, (int, float)):
                print(f"DownsideProb:       {ddp:.2f}")
            if isinstance(crash, (int, float)):
                print(f"CrashProb:          {crash:.2f}")

    if rec:
        bias = rec.get("bias")
        center = rec.get("centerPct")
        lower = rec.get("lowerPct")
        upper = rec.get("upperPct")
        step = rec.get("suggestedStepPct")
        levels = rec.get("levels")
        comment = rec.get("comment")

        parts = []
        if bias:
            parts.append(f"bias={bias}")
        if isinstance(levels, int):
            parts.append(f"levels={levels}")
        if isinstance(step, (int, float)):
            parts.append(f"step≈{step:.4f}%")
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            parts.append(f"range=[{lower:+.4f}%, {upper:+.4f}%]")
        if isinstance(center, (int, float)):
            parts.append(f"center={center:+.4f}%")

        if parts:
            print("Grid recommendation: " + "  ".join(parts))
        if comment:
            print(f"Note: {comment}")

    if isinstance(price_ctx, dict) and price_ctx:
        zone = price_ctx.get("zone")
        sigma_offset = price_ctx.get("sigmaOffset")
        if zone is not None:
            if isinstance(sigma_offset, (int, float)):
                print(f"Price context: {zone} (sigmaOffset={sigma_offset:+.3f})")
            else:
                print(f"Price context: {zone}")

    if isinstance(grid_levels, list) and grid_levels:
        # Print compact sigma ladder.
        ladder = []
        for gl in grid_levels:
            if not isinstance(gl, dict):
                continue
            label = gl.get("label")
            pct = gl.get("pct")
            if label and isinstance(pct, (int, float)):
                ladder.append(f"{label}:{pct:+.3f}%")
        if ladder:
            print("Grid levels: " + "  ".join(ladder))

    if isinstance(sigma_levels, list) and sigma_levels:
        # Show first 1σ level hit probabilities (most actionable).
        one = next(
            (x for x in sigma_levels if isinstance(x, dict) and x.get("k") == 1), None
        )
        if isinstance(one, dict):
            tp = one.get("tpHitProb")
            sl = one.get("slHitProb")
            if isinstance(tp, (int, float)) or isinstance(sl, (int, float)):
                tp_s = f"{tp:.2f}" if isinstance(tp, (int, float)) else "n/a"
                sl_s = f"{sl:.2f}" if isinstance(sl, (int, float)) else "n/a"
                print(f"1σ hit probs: tpHit={tp_s}  slHit={sl_s}")

    if isinstance(terminal, dict) and terminal:
        p05 = terminal.get("p05")
        p95 = terminal.get("p95")
        if isinstance(p05, (int, float)) and isinstance(p95, (int, float)):
            print(f"Terminal percentiles: p05={p05:+.3f}%  p95={p95:+.3f}%")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_recalc(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.recalc_patterns(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        q=args.q,
        f=args.f,
        limit=args.limit,
        sort=args.sort,
        force=args.force,
    )

    matches = res.get("matches") or []

    print("=" * 78)
    print("HISTORICAL REPLAY (RECALC)")
    print("=" * 78)
    print(
        f"Symbol: {args.symbol}  Interval: {args.interval}  start={args.start}  q={args.q}  f={args.f}  limit={args.limit}"
    )
    print(f"Matches: {len(matches)}")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_batch(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)

    try:
        reqs = json.loads(args.requests_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON for --requests-json: {e}")

    if not isinstance(reqs, list) or not all(isinstance(x, dict) for x in reqs):
        raise SystemExit("--requests-json must be a JSON array of objects")

    res = client.batch_search(reqs)

    print("=" * 78)
    print("BATCH SEARCH")
    print("=" * 78)
    print(f"Requests: {len(reqs)}")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))
    else:
        results = res.get("results") if isinstance(res, dict) else None
        if isinstance(results, list):
            for idx, r in enumerate(results, start=1):
                matches = (r or {}).get("matches") or []
                symbol = _safe_get(r or {}, ["meta", "symbol"], None) or (
                    reqs[idx - 1].get("symbol")
                )
                interval = _safe_get(r or {}, ["meta", "interval"], None) or (
                    reqs[idx - 1].get("interval")
                )
                print(f"  #{idx:02d} {symbol} {interval}: {len(matches)} matches")

    return 0


def _backtest_period_start(days: int) -> int:
    return int((time.time() - days * 24 * 60 * 60) * 1000)


def _fmt_pct(v: Any, *, digits: int = 2) -> str:
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:+.{digits}f}%"


def _print_backtest_report_card(
    stats: Dict[str, Any], *, include_annualized: bool = True
) -> None:
    if not isinstance(stats, dict) or not stats:
        return

    # Keep this as a best-effort printer; API may evolve.
    total_return = stats.get("totalReturnPct")
    annualized = stats.get("annualizedReturnPct")
    win_rate = stats.get("winRate")
    profit_factor = stats.get("profitFactor")
    max_dd = stats.get("maxDrawdownPct")
    sharpe = stats.get("sharpeRatio")
    sortino = stats.get("sortinoRatio")
    calmar = stats.get("calmarRatio")
    total_trades = stats.get("totalTrades")
    avg_win = stats.get("avgWinPct")
    avg_loss = stats.get("avgLossPct")

    print("Report card:")
    if isinstance(total_trades, (int, float)):
        print(f"  totalTrades:     {int(total_trades)}")
    if isinstance(win_rate, (int, float)):
        print(f"  winRate:         {win_rate:.1f}%")
    if isinstance(profit_factor, (int, float)):
        print(f"  profitFactor:    {profit_factor:.3f}")
    if isinstance(total_return, (int, float)):
        print(f"  totalReturnPct:  {total_return:+.2f}%")
    if include_annualized and isinstance(annualized, (int, float)):
        print(f"  annualizedPct:   {annualized:+.2f}%")
    if isinstance(max_dd, (int, float)):
        print(f"  maxDrawdownPct:  {max_dd:.2f}%")
    if include_annualized and isinstance(sharpe, (int, float)):
        print(f"  sharpeRatio:     {sharpe:.3f}")
    if include_annualized and isinstance(sortino, (int, float)):
        print(f"  sortinoRatio:    {sortino:.3f}")
    if include_annualized and isinstance(calmar, (int, float)):
        print(f"  calmarRatio:     {calmar:.3f}")
    if isinstance(avg_win, (int, float)):
        print(f"  avgWinPct:       {avg_win:+.2f}%")
    if isinstance(avg_loss, (int, float)):
        print(f"  avgLossPct:      {avg_loss:+.2f}%")

    if not include_annualized and any(
        isinstance(x, (int, float)) for x in (annualized, sharpe, sortino, calmar)
    ):
        print("  note: annualized metrics hidden (micro-validation window)")


def _extract_trades(backtest_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    trades = backtest_result.get("results")
    if isinstance(trades, list):
        return trades
    trades = backtest_result.get("trades")
    return trades if isinstance(trades, list) else []


def _parse_csv_list(value: str) -> List[str]:
    items = [v.strip() for v in (value or "").split(",")]
    return [v for v in items if v]


def _grid_hint_from_grid_stats(grid_stats: Dict[str, Any]) -> Optional[str]:
    regime_label = _safe_get(grid_stats, ["regime", "label"], None)
    if not isinstance(regime_label, str) or not regime_label.strip():
        return None

    r = regime_label.upper()
    if "MEAN_REVERSION" in r or "RANGE" in r:
        return "GRID_FRIENDLY"
    if "MOMENTUM" in r or "BREAKOUT" in r:
        return "TREND_RISK"
    if "VOLATILITY" in r:
        return "HIGH_VOL"
    if "DOWNTREND" in r or "UPTREND" in r:
        return "TRENDING"

    return "MIXED"


def _grid_hint_pretty(code: Optional[str]) -> str:
    if not code:
        return ""

    mapping = {
        "GRID_FRIENDLY": "Grid-friendly",
        "TREND_RISK": "Trend-only",
        "HIGH_VOL": "High-vol (widen steps)",
        "TRENDING": "Trending (bias-aware)",
        "MIXED": "Mixed",
    }
    return mapping.get(code, str(code))


def _forecast_pct_from_search_result(search_result: Dict[str, Any]) -> Optional[float]:
    forecast = (
        search_result.get("forecast") if isinstance(search_result, dict) else None
    )
    if not isinstance(forecast, dict):
        return None
    median = forecast.get("median")
    if not isinstance(median, list) or len(median) < 2:
        return None
    start_price = median[0]
    end_price = median[-1]
    if not isinstance(start_price, (int, float)) or not isinstance(
        end_price, (int, float)
    ):
        return None
    if start_price == 0:
        return None
    return (end_price - start_price) / start_price * 100


def _stability_check_via_recalc(
    client: Client,
    *,
    symbol: str,
    interval: str,
    base_offset: int,
    offsets: List[int],
    q: int,
    f: int,
    limit: int,
    sort: str,
    force: bool,
    threshold: float,
) -> Dict[str, Any]:
    scores: List[float] = []

    for drift in offsets:
        check_offset = base_offset + drift
        try:
            res = client.recalc_patterns(
                symbol=symbol,
                interval=interval,
                start=check_offset,
                q=q,
                f=f,
                limit=limit,
                sort=sort,
                force=force,
            )
            pct = _forecast_pct_from_search_result(res)
            if isinstance(pct, (int, float)):
                scores.append(float(pct))
        except Exception:
            continue

    if not scores:
        return {
            "stable": False,
            "volatility": None,
            "mean_forecast": None,
            "min_forecast": None,
            "max_forecast": None,
        }

    volatility = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    mean_forecast = sum(scores) / len(scores)
    return {
        "stable": volatility < threshold,
        "volatility": float(volatility),
        "mean_forecast": float(mean_forecast),
        "min_forecast": float(min(scores)),
        "max_forecast": float(max(scores)),
    }


def cmd_backtest(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    include_stats = not bool(getattr(args, "no_stats", False))
    start_ts = (
        args.start_ts
        if args.start_ts is not None
        else _backtest_period_start(args.days)
    )
    end_ts = args.end_ts

    fee_pct = (
        args.fee_pct
        if args.fee_pct is not None
        else _env_float("AIPP_FEE_PCT", "AIPP_DEMO_FEE_PCT")
    )
    slippage_pct = (
        args.slippage_pct
        if args.slippage_pct is not None
        else _env_float("AIPP_SLIPPAGE_PCT", "AIPP_DEMO_SLIPPAGE_PCT")
    )
    fee_pct = float(fee_pct) if isinstance(fee_pct, (int, float)) else 0.0
    slippage_pct = (
        float(slippage_pct) if isinstance(slippage_pct, (int, float)) else 0.0
    )

    res = client.backtest(
        symbol=args.symbol,
        interval=args.interval,
        q=args.q,
        f=args.f,
        step=args.step,
        top_k=args.top_k,
        min_prob=args.min_prob,
        start_ts=start_ts,
        end_ts=end_ts,
        include_stats=include_stats,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )

    trades = _extract_trades(res)
    returns = [t.get("actualReturnPct", 0.0) for t in trades if isinstance(t, dict)]
    wins = [r for r in returns if isinstance(r, (int, float)) and r > 0]
    losses = [r for r in returns if isinstance(r, (int, float)) and r < 0]

    print("=" * 78)
    print("WALK-FORWARD BACKTEST")
    print("=" * 78)
    print(
        f"Symbol: {args.symbol}  Interval: {args.interval}  q={args.q}  f={args.f}  step={args.step}  minProb={args.min_prob}"
    )
    if args.start_ts is not None or args.end_ts is not None:
        s = _fmt_dt(start_ts) if isinstance(start_ts, int) else "n/a"
        e = _fmt_dt(end_ts) if isinstance(end_ts, int) else "now"
        print(f"Period: {s} → {e}")
    else:
        print(f"Period: last {args.days} days (start {_fmt_dt(start_ts)})")
    if fee_pct != 0.0 or slippage_pct != 0.0:
        print(f"Costs: feePct={fee_pct:.4f}%  slippagePct={slippage_pct:.4f}%")

    print(f"Signals: {len(trades)}")
    if trades:
        win_rate = (len(wins) / len(trades) * 100) if len(trades) else 0.0
        total_return = sum(r for r in returns if isinstance(r, (int, float)))
        worst = min(losses) if losses else 0.0
        best = max(wins) if wins else 0.0
        print(f"Win Rate:     {win_rate:.1f}%  (wins/losses {len(wins)}/{len(losses)})")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Best/Worst:   {best:+.2f}% / {worst:+.2f}%")

        print("Samples:")
        for t in trades[: min(5, len(trades))]:
            ts = t.get("ts")
            ret = t.get("actualReturnPct")
            dt = _fmt_dt(ts) if isinstance(ts, int) else "n/a"
            if isinstance(ret, (int, float)):
                tag = "WIN" if ret > 0 else "LOSS" if ret < 0 else "FLAT"
                print(f"  {dt}  return={ret:+.2f}%  {tag}")
            else:
                print(f"  {dt}  return=n/a")

    stats = res.get("stats") if isinstance(res, dict) else None
    if isinstance(stats, dict) and stats:
        print("-")
        _print_backtest_report_card(stats)

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_backtest_specific(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)

    if args.timestamp is None and args.offset is None:
        raise SystemExit("Provide one of: --timestamp or --offset")
    if args.timestamp is not None and args.offset is not None:
        raise SystemExit("Provide only one of: --timestamp or --offset")

    fee_pct = (
        args.fee_pct
        if args.fee_pct is not None
        else _env_float("AIPP_FEE_PCT", "AIPP_BT_FEE_PCT")
    )
    slippage_pct = (
        args.slippage_pct
        if args.slippage_pct is not None
        else _env_float("AIPP_SLIPPAGE_PCT", "AIPP_BT_SLIPPAGE_PCT")
    )
    fee_pct = float(fee_pct) if isinstance(fee_pct, (int, float)) else 0.04
    slippage_pct = (
        float(slippage_pct) if isinstance(slippage_pct, (int, float)) else 0.02
    )

    res = client.backtest_specific_pattern(
        symbol=args.symbol,
        interval=args.interval,
        q=args.q,
        f=args.f,
        timestamp=args.timestamp,
        offset=args.offset,
        top_k=args.top_k,
        include_stats=not args.no_stats,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )

    print("=" * 78)
    print("SPECIFIC PATTERN BACKTEST (MICRO-VALIDATION)")
    print("=" * 78)
    print(
        f"Symbol: {args.symbol}  Interval: {args.interval}  q={args.q}  f={args.f}  topK={args.top_k}"
    )
    if args.timestamp is not None:
        print(f"Timestamp: {args.timestamp} ({_fmt_dt(args.timestamp)})")
    if args.offset is not None:
        print(f"Offset: {args.offset}")
    print(f"Costs: feePct={fee_pct:.4f}%  slippagePct={slippage_pct:.4f}%")

    avg_return = res.get("avgReturn") if isinstance(res, dict) else None
    win_rate = res.get("winRate") if isinstance(res, dict) else None
    trades_taken = res.get("tradesTaken") if isinstance(res, dict) else None
    total_occ = res.get("totalOccurrences") if isinstance(res, dict) else None
    if avg_return is not None:
        print(f"Avg return: {_fmt_pct(avg_return, digits=2)}")
    if isinstance(win_rate, (int, float)):
        print(f"Win rate:  {win_rate:.1f}%")
    if isinstance(trades_taken, (int, float)):
        print(f"Trades:    {int(trades_taken)} (occurrences={total_occ})")

    events = res.get("events") if isinstance(res, dict) else None
    if isinstance(events, list) and events:
        e0 = events[0] if isinstance(events[0], dict) else {}
        signal = e0.get("signal")
        outcome = e0.get("outcomePct")
        sim = e0.get("similarity")
        print("Sample event:")
        if signal is not None:
            print(f"  signal:   {signal}")
        if isinstance(outcome, (int, float)):
            print(f"  outcome:  {outcome:+.2f}%")
        if isinstance(sim, (int, float)):
            print(f"  similarity: {sim:.4f}")

    stats = res.get("stats") if isinstance(res, dict) else None
    if isinstance(stats, dict) and stats:
        print("-")
        _print_backtest_report_card(stats, include_annualized=False)

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_rl_episodes(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)

    if args.anchor_ts is None and args.current_state is None:
        raise SystemExit("Provide one of: --anchor-ts or --current-state")
    if args.anchor_ts is not None and args.current_state is not None:
        raise SystemExit("Provide only one of: --anchor-ts or --current-state")

    current_state = (
        _parse_float_list(args.current_state)
        if isinstance(args.current_state, str)
        else None
    )

    res = client.get_rl_episodes(
        symbol=args.symbol,
        interval=args.interval,
        current_state=current_state,
        anchor_ts=args.anchor_ts,
        forecast_horizon=args.forecast_horizon,
        num_episodes=args.num_episodes,
        min_similarity=args.min_similarity,
        include_actions=bool(args.include_actions),
        reward_type=args.reward_type,
        sampling_strategy=args.sampling_strategy,
    )

    meta = res.get("meta") if isinstance(res, dict) else None
    episodes = res.get("episodes") if isinstance(res, dict) else None
    stats = res.get("statistics") if isinstance(res, dict) else None
    risk = res.get("riskAnalysis") if isinstance(res, dict) else None

    print("=" * 78)
    print("RL EPISODES (PARALLEL UNIVERSES)")
    print("=" * 78)
    print(f"Base URL: {common.base_url}")
    print(
        f"Symbol: {args.symbol}  Interval: {args.interval}  horizon={args.forecast_horizon}  numEpisodes={args.num_episodes}  minSim={args.min_similarity:.2f}"
    )
    if args.anchor_ts is not None:
        print(f"AnchorTs: {args.anchor_ts} ({_fmt_dt(args.anchor_ts)})")
    if current_state is not None:
        print(f"CurrentState: {len(current_state)} dims")

    if isinstance(meta, dict) and meta:
        rt = meta.get("regimeType")
        rc = meta.get("regimeConfidence")
        te = meta.get("totalEpisodes")
        line = []
        if te is not None:
            line.append(f"totalEpisodes={te}")
        if rt is not None:
            if isinstance(rc, (int, float)):
                line.append(f"regime={rt} (conf={rc:.2f})")
            else:
                line.append(f"regime={rt}")
        if line:
            print("Meta: " + "  ".join(line))

    if isinstance(episodes, list):
        print(f"Episodes returned: {len(episodes)}")
        sims = [
            e.get("similarity")
            for e in episodes
            if isinstance(e, dict) and isinstance(e.get("similarity"), (int, float))
        ]
        if sims:
            print(
                f"Similarity: min={min(sims):.3f}  mean={sum(sims) / len(sims):.3f}  max={max(sims):.3f}"
            )
        e0 = episodes[0] if episodes and isinstance(episodes[0], dict) else None
        if isinstance(e0, dict):
            tl = e0.get("transitions")
            if isinstance(tl, list):
                print(f"Sample episode transitions: {len(tl)}")

    if isinstance(stats, dict) and stats:
        print("-")
        print("Statistics:")
        for k in ("avgReturn", "winRate", "avgMaxDrawdown"):
            if k in stats:
                print(f"  {k}: {stats.get(k)}")

    if isinstance(risk, dict) and risk:
        print("-")
        print("Risk analysis:")
        for k in (
            "volatilityForecast",
            "valueAtRisk95",
            "downsideProbability",
            "crashProbability",
            "kellyFraction",
        ):
            if k in risk:
                print(f"  {k}: {risk.get(k)}")

    print("-")
    print("How to use:")
    print("  Train a specialist RL agent on these episodes (context-aware RL).")
    print("  If episodes are empty: lower minSimilarity or change anchor/context.")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_rl_training_batch(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)

    res = client.get_rl_training_batch(
        symbol=args.symbol,
        interval=args.interval,
        query_length=args.query_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        min_similarity=args.min_similarity,
    )

    meta = res.get("meta") if isinstance(res, dict) else None
    data = res.get("data") if isinstance(res, dict) else None

    print("=" * 78)
    print("RL TRAINING BATCH (TENSOR FACTORY)")
    print("=" * 78)
    print(f"Base URL: {common.base_url}")
    print(
        f"Symbol: {args.symbol}  Interval: {args.interval}  queryLength={args.query_length}  horizon={args.forecast_horizon}  batchSize={args.batch_size}  minSim={args.min_similarity:.2f}"
    )

    if isinstance(meta, dict) and meta:
        print(f"Meta: {meta}")

    if isinstance(data, dict) and data:
        states = data.get("states")
        next_states = data.get("nextStates")
        rewards = data.get("rewards")
        dones = data.get("dones")

        def _n(x: Any) -> str:
            return str(len(x)) if isinstance(x, list) else "n/a"

        print("-")
        print("Tuple sizes (flat arrays):")
        print(f"  states:      {_n(states)}")
        print(f"  nextStates:  {_n(next_states)}")
        print(f"  rewards:     {_n(rewards)}")
        print(f"  dones:       {_n(dones)}")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump(res))

    return 0


def cmd_dataset_status(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_dataset_status(symbol=args.symbol)
    print(_json_dump(res))
    return 0


def cmd_dataset_expand(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.expand_dataset(
        symbol=args.symbol,
        interval=args.interval,
        bars=args.bars,
        since=args.since,
    )
    print(_json_dump(res))
    return 0


def cmd_dataset_delete(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.delete_dataset(
        symbol=args.symbol,
        interval=args.interval,
        from_ts=args.from_ts,
        to_ts=args.to_ts,
    )
    print(_json_dump(res))
    return 0


def cmd_dataset_stats(common: CommonArgs, _args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_dataset_stats()
    print(_json_dump(res))
    return 0


def cmd_dataset_gaps(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_dataset_gaps(symbol=args.symbol, interval=args.interval)
    print(_json_dump(res))
    return 0


def cmd_dataset_vectors(common: CommonArgs, _args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_dataset_vectors()
    print(_json_dump(res))
    return 0


def cmd_regime_catalog(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_rl_regimes(symbol=args.symbol, interval=args.interval)
    print(_json_dump(res))
    return 0


def cmd_regime_detect(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    current_state = (
        _parse_float_list(args.current_state) if args.current_state else None
    )
    res = client.detect_regime(
        symbol=args.symbol,
        interval=args.interval,
        query_length=args.query_length,
        timestamp=args.timestamp,
        current_state=current_state,
    )
    print(_json_dump(res))
    return 0


def cmd_regime_latest(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_current_regime(
        symbol=args.symbol, interval=args.interval, query_length=args.query_length
    )
    print(_json_dump(res))
    return 0


def cmd_ann_status(common: CommonArgs, _args: argparse.Namespace) -> int:
    client = _make_client(common)
    res = client.get_ann_status()
    print(_json_dump(res))
    return 0


def cmd_ann_search(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    vector = _parse_float_list(args.vector)
    res = client.ann_search(vector=vector, k=args.k, ef=args.ef)
    print(_json_dump(res))
    return 0


def cmd_ann_upsert(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    vector = _parse_float_list(args.vector)
    res = client.ann_upsert(
        id=args.id,
        vector=vector,
        symbol=args.symbol,
        interval=args.interval,
    )
    print(_json_dump(res))
    return 0


def cmd_audit(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)
    start_ts = _backtest_period_start(args.days)

    fee_pct = (
        args.fee_pct
        if args.fee_pct is not None
        else _env_float("AIPP_FEE_PCT", "AIPP_DEMO_FEE_PCT")
    )
    slippage_pct = (
        args.slippage_pct
        if args.slippage_pct is not None
        else _env_float("AIPP_SLIPPAGE_PCT", "AIPP_DEMO_SLIPPAGE_PCT")
    )
    fee_pct = float(fee_pct) if isinstance(fee_pct, (int, float)) else 0.0
    slippage_pct = (
        float(slippage_pct) if isinstance(slippage_pct, (int, float)) else 0.0
    )

    backtest_res = client.backtest(
        symbol=args.symbol,
        interval=args.interval,
        q=args.q,
        f=args.f,
        step=args.step,
        top_k=args.top_k,
        min_prob=args.min_prob,
        start_ts=start_ts,
        include_stats=True,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )

    trades = _extract_trades(backtest_res)

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

    print("=" * 78)
    print("REGIME ATTRIBUTION (LOSSES)")
    print("=" * 78)
    print(
        f"Symbol: {args.symbol}  Interval: {args.interval}  q={args.q}  f={args.f}  step={args.step}  minProb={args.min_prob}"
    )
    print(f"Period: last {args.days} days (start {_fmt_dt(start_ts)})")

    report = auditor.analyze_losses(args.symbol, args.interval, trades_for_audit)
    total_losses = report.get("total_losses", 0)
    dist = report.get("regime_distribution", {}) or {}

    print(f"Total losses analyzed: {total_losses}")
    if isinstance(dist, dict) and dist:
        sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        print("Losses by regime:")
        for rid, count in sorted_items:
            pct = (count / total_losses * 100) if total_losses else 0
            print(f"  {rid:20s}: {count:3d} ({pct:4.1f}%)")
        worst_regime = sorted_items[0][0]
        print(f"\nAvoid trading in: {worst_regime}")
    else:
        print("No losing trades found in this sample.")

    if args.raw:
        print("\nRAW JSON")
        print(_json_dump({"backtest": backtest_res, "audit": report}))

    return 0


def cmd_scan(common: CommonArgs, args: argparse.Namespace) -> int:
    client = _make_client(common)

    symbols = _parse_csv_list(args.symbols)
    if not symbols:
        raise SystemExit("--symbols must be a comma-separated list")

    blocked_regimes = set(_parse_csv_list(args.block_regimes))
    stability_offsets = [int(x) for x in _parse_csv_list(args.stability_offsets)]
    if 0 not in stability_offsets:
        stability_offsets = [0] + stability_offsets

    def run_once() -> int:
        reqs: List[Dict[str, Any]] = []
        for sym in symbols:
            reqs.append(
                {
                    "symbol": sym,
                    "interval": args.interval,
                    "q": args.q,
                    "f": args.f,
                    "limit": args.limit,
                    "sort": args.sort,
                }
            )

        batch_res = client.batch_search(reqs)
        batch_results = (
            batch_res.get("results") if isinstance(batch_res, dict) else None
        )
        if not isinstance(batch_results, list) or len(batch_results) != len(symbols):
            batch_results = [{} for _ in symbols]

        rows: List[Dict[str, Any]] = []
        for idx, sym in enumerate(symbols):
            search_res = batch_results[idx] if idx < len(batch_results) else {}
            search_res = search_res if isinstance(search_res, dict) else {}

            matches = search_res.get("matches") or []
            match_count = len(matches) if isinstance(matches, list) else 0
            forecast_pct = _forecast_pct_from_search_result(search_res)

            regime_id = None
            regime_error = None
            try:
                r = client.get_current_regime(sym, args.interval)
                regime_id = _safe_get(r, ["currentRegime", "id"], None)
            except Exception as e:
                regime_error = str(e)

            stability = None
            if not args.no_stability:
                stability = _stability_check_via_recalc(
                    client,
                    symbol=sym,
                    interval=args.interval,
                    base_offset=0,
                    offsets=stability_offsets,
                    q=args.q,
                    f=args.f,
                    limit=args.limit,
                    sort=args.sort,
                    force=args.force,
                    threshold=args.stability_threshold,
                )

            decision = "NO"
            reason = ""
            if regime_error:
                reason = f"regime_error:{regime_error}"
            elif regime_id and regime_id in blocked_regimes:
                reason = f"blocked_regime:{regime_id}"
            elif match_count == 0:
                reason = "no_matches"
            elif stability is not None and stability.get("volatility") is None:
                reason = "no_forecast_for_stability"
            elif stability is not None and not stability.get("stable", False):
                vol = stability.get("volatility")
                reason = (
                    f"unstable(vol={vol:.3f})"
                    if isinstance(vol, (int, float))
                    else "unstable"
                )
            else:
                decision = "GO"
                reason = "ok"

            grid_hint: Optional[str] = None
            if args.grid_hint and decision == "GO":
                try:
                    g = client.get_grid_stats(sym, args.interval)
                    grid_hint = _grid_hint_from_grid_stats(g)
                except Exception:
                    grid_hint = None

            rows.append(
                {
                    "symbol": sym,
                    "interval": args.interval,
                    "regime": regime_id,
                    "matches": match_count,
                    "forecastPct": forecast_pct,
                    "stable": None
                    if stability is None
                    else bool(stability.get("stable")),
                    "volatility": None
                    if stability is None
                    else stability.get("volatility"),
                    "gridHint": grid_hint,
                    "decision": decision,
                    "reason": reason,
                }
            )

        if args.json:
            print(_json_dump({"blockedRegimes": sorted(blocked_regimes), "rows": rows}))
            return 0

        if args.watch:
            print("\n" + _fmt_dt(int(time.time() * 1000)))

        print("=" * 78)
        print("WATCHLIST SCAN (GO/NO-GO)")
        print("=" * 78)
        print(f"Base URL: {common.base_url}")
        print(
            f"Interval: {args.interval}  q={args.q}  f={args.f}  limit={args.limit}  sort={args.sort}  stability_threshold={args.stability_threshold}"
        )
        if blocked_regimes:
            print(f"Blocked regimes: {', '.join(sorted(blocked_regimes))}")
        print(
            "Stability: DISABLED"
            if args.no_stability
            else f"Stability offsets: {stability_offsets}"
        )
        print("-")

        header = f"{'SYMBOL':10s} {'REGIME':18s} {'M':>3s} {'FCST%':>7s} {'VOL':>7s} {'STB':>3s} {'DEC':>3s}"
        if args.grid_hint:
            header += "  GRID_HINT"
        header += "  REASON"
        print(header)
        print("-" * len(header))

        for r in rows:
            sym = (r.get("symbol") or "")[:10]
            regime = (r.get("regime") or "N/A")[:18]
            m = int(r.get("matches") or 0)
            fc = r.get("forecastPct")
            fc_s = f"{fc:+.2f}" if isinstance(fc, (int, float)) else "  n/a"
            vol = r.get("volatility")
            vol_s = f"{vol:7.3f}" if isinstance(vol, (int, float)) else "   n/a"
            stb = r.get("stable")
            stb_s = "YES" if stb is True else "NO" if stb is False else "n/a"
            dec = r.get("decision") or "NO"
            gh = r.get("gridHint")
            if args.grid_hint:
                gh_val = (
                    _grid_hint_pretty(gh)
                    if args.grid_hint_human
                    else (str(gh) if gh else "")
                )
                gh_s = gh_val.ljust(22)
            else:
                gh_s = ""
            reason = r.get("reason") or ""
            print(
                f"{sym:10s} {regime:18s} {m:3d} {fc_s:>7s} {vol_s:>7s} {stb_s:>3s} {dec:>3s}"
                + (f"  {gh_s}" if args.grid_hint else "")
                + f"  {reason}"
            )

        go = sum(1 for r in rows if r.get("decision") == "GO")
        print("-")
        print(f"GO: {go}/{len(rows)}")
        return 0

    if not args.watch:
        return run_once()

    try:
        while True:
            run_once()
            time.sleep(max(1, int(args.every)))
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aipp",
        description="Hedge-fund oriented CLI for AI Price Patterns (matches UI /api-docs/patterns).",
    )

    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default {DEFAULT_BASE_URL})",
    )
    p.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key (optional; can set AIPP_API_KEY)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="GET /api/patterns")
    s.add_argument("--symbol", required=True)
    s.add_argument("--interval", default="1h")
    s.add_argument("--q", type=int, default=60)
    s.add_argument("--f", type=int, default=30)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--sort", default="similarity")
    s.add_argument("--force", action="store_true")
    s.add_argument(
        "--cross-asset", action="store_true", help="Search across all symbols"
    )
    s.add_argument("--raw", action="store_true")
    s.set_defaults(func=cmd_search)

    sig = sub.add_parser("signals", help="GET /api/patterns/signals")
    sig.add_argument("--raw", action="store_true")
    sig.set_defaults(func=cmd_signals)

    # --- Dataset Group ---
    ds = sub.add_parser("dataset", help="Dataset management")
    ds_sub = ds.add_subparsers(dest="subcmd", required=True)

    dst = ds_sub.add_parser("status", help="Get dataset status")
    dst.add_argument("--symbol", help="Optional symbol filter")
    dst.set_defaults(func=cmd_dataset_status)

    dse = ds_sub.add_parser("expand", help="Expand dataset")
    dse.add_argument("--symbol", required=True)
    dse.add_argument("--interval", required=True)
    dse.add_argument("--bars", type=int)
    dse.add_argument("--since", type=int)
    dse.set_defaults(func=cmd_dataset_expand)

    dsd = ds_sub.add_parser("delete", help="Delete dataset range")
    dsd.add_argument("--symbol", required=True)
    dsd.add_argument("--interval", required=True)
    dsd.add_argument("--from-ts", type=int)
    dsd.add_argument("--to-ts", type=int)
    dsd.set_defaults(func=cmd_dataset_delete)

    dsts = ds_sub.add_parser("stats", help="Get dataset stats")
    dsts.set_defaults(func=cmd_dataset_stats)

    dsg = ds_sub.add_parser("gaps", help="Detect dataset gaps")
    dsg.add_argument("--symbol", required=True)
    dsg.add_argument("--interval", required=True)
    dsg.set_defaults(func=cmd_dataset_gaps)

    dsv = ds_sub.add_parser("vectors", help="List vector datasets")
    dsv.set_defaults(func=cmd_dataset_vectors)

    # --- Regime Group ---
    reg = sub.add_parser("regime", help="Market regime analysis")
    reg_sub = reg.add_subparsers(dest="subcmd", required=True)

    rc = reg_sub.add_parser("catalog", help="List all regimes")
    rc.add_argument("--symbol", default="BTCUSDT")
    rc.add_argument("--interval", default="1h")
    rc.set_defaults(func=cmd_regime_catalog)

    rd = reg_sub.add_parser("detect", help="Detect regime for state/ts")
    rd.add_argument("--symbol", default="BTCUSDT")
    rd.add_argument("--interval", default="1h")
    rd.add_argument("--query-length", type=int, default=40)
    rd.add_argument("--timestamp", type=int)
    rd.add_argument("--current-state", help="JSON array or CSV")
    rd.set_defaults(func=cmd_regime_detect)

    rl = reg_sub.add_parser("latest", help="Get current regime")
    rl.add_argument("--symbol", default="BTCUSDT")
    rl.add_argument("--interval", default="1h")
    rl.add_argument("--query-length", type=int, default=40)
    rl.set_defaults(func=cmd_regime_latest)

    # --- ANN Group ---
    ann = sub.add_parser("ann", help="ANN index operations")
    ann_sub = ann.add_subparsers(dest="subcmd", required=True)

    as1 = ann_sub.add_parser("status", help="Get ANN status")
    as1.set_defaults(func=cmd_ann_status)

    as2 = ann_sub.add_parser("search", help="Search ANN index")
    as2.add_argument("--vector", required=True, help="JSON array or CSV")
    as2.add_argument("--k", type=int, default=10)
    as2.add_argument("--ef", type=int, default=64)
    as2.set_defaults(func=cmd_ann_search)

    au = ann_sub.add_parser("upsert", help="Upsert to ANN index")
    au.add_argument("--id", type=int, required=True)
    au.add_argument("--vector", required=True, help="JSON array or CSV")
    au.add_argument("--symbol", required=True)
    au.add_argument("--interval", required=True)
    au.set_defaults(func=cmd_ann_upsert)

    m = sub.add_parser("metrics", help="GET /api/patterns/metrics")
    m.add_argument("--symbol", required=True)
    m.add_argument("--interval", default="1h")
    m.add_argument("--q", type=int, default=40)
    m.add_argument("--f", type=int, default=30)
    m.add_argument("--raw", action="store_true")
    m.set_defaults(func=cmd_metrics)

    g = sub.add_parser("grid", help="GET /api/patterns/grid")
    g.add_argument("--symbol", required=True)
    g.add_argument("--interval", default="1h")
    g.add_argument("--raw", action="store_true")
    g.set_defaults(func=cmd_grid)

    r = sub.add_parser("recalc", help="GET /api/patterns-recalc")
    r.add_argument("--symbol", required=True)
    r.add_argument("--interval", default="1h")
    r.add_argument(
        "--start", type=int, required=True, help="Historical offset (integer)"
    )
    r.add_argument("--q", type=int, default=40)
    r.add_argument("--f", type=int, default=30)
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--sort", default="similarity")
    r.add_argument("--force", action="store_true")
    r.add_argument("--raw", action="store_true")
    r.set_defaults(func=cmd_recalc)

    b = sub.add_parser("batch", help="POST /api/patterns/batch")
    b.add_argument(
        "--requests-json",
        required=True,
        help='JSON array of request objects, e.g. [{"symbol":"BTCUSDT","interval":"1h","q":60,"f":30,"limit":16}]',
    )
    b.add_argument("--raw", action="store_true")
    b.set_defaults(func=cmd_batch)

    bt = sub.add_parser("backtest", help="GET /api/patterns/backtest")
    bt.add_argument("--symbol", required=True)
    bt.add_argument("--interval", default="1h")
    bt.add_argument("--days", type=int, default=90)
    bt.add_argument(
        "--start-ts",
        type=int,
        default=None,
        help="Override start timestamp in ms (takes precedence over --days)",
    )
    bt.add_argument(
        "--end-ts",
        type=int,
        default=None,
        help="Optional end timestamp in ms",
    )
    bt.add_argument("--q", type=int, default=60)
    bt.add_argument("--f", type=int, default=24)
    bt.add_argument("--step", type=int, default=24)
    bt.add_argument("--top-k", type=int, default=5)
    bt.add_argument("--min-prob", type=float, default=0.50)
    bt.add_argument(
        "--fee-pct",
        type=float,
        default=None,
        help="Exchange fee in percent (e.g. 0.04)",
    )
    bt.add_argument(
        "--slippage-pct",
        type=float,
        default=None,
        help="Execution slippage in percent (e.g. 0.02)",
    )
    bt.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable stats/equity curve computation (faster)",
    )
    bt.add_argument("--raw", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    bts = sub.add_parser(
        "backtest-specific", help="POST /api/patterns/backtest/specific"
    )
    bts.add_argument("--symbol", required=True)
    bts.add_argument("--interval", default="1h")
    bts.add_argument("--q", type=int, default=24)
    bts.add_argument("--f", type=int, default=12)
    bts.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Pattern timestamp in ms (recommended)",
    )
    bts.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Dataset offset (alternative to --timestamp)",
    )
    bts.add_argument("--top-k", type=int, default=5)
    bts.add_argument(
        "--fee-pct",
        type=float,
        default=None,
        help="Exchange fee in percent (e.g. 0.04)",
    )
    bts.add_argument(
        "--slippage-pct",
        type=float,
        default=None,
        help="Execution slippage in percent (e.g. 0.02)",
    )
    bts.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable stats/equity curve computation",
    )
    bts.add_argument("--raw", action="store_true")
    bts.set_defaults(func=cmd_backtest_specific)

    rle = sub.add_parser("rl-episodes", help="POST /api/rl/episodes")
    rle.add_argument("--symbol", default="BTCUSDT")
    rle.add_argument("--interval", default="1h")
    rle.add_argument(
        "--anchor-ts",
        type=int,
        default=None,
        help="Anchor timestamp in ms (recommended)",
    )
    rle.add_argument(
        "--current-state",
        default=None,
        help='State vector (JSON array like "[1,2,3]" or comma-separated floats)',
    )
    rle.add_argument("--forecast-horizon", type=int, default=24)
    rle.add_argument("--num-episodes", type=int, default=50)
    rle.add_argument("--min-similarity", type=float, default=0.80)
    rle.add_argument("--include-actions", action="store_true")
    rle.add_argument("--reward-type", default="returns")
    rle.add_argument("--sampling-strategy", default="uniform")
    rle.add_argument("--raw", action="store_true")
    rle.set_defaults(func=cmd_rl_episodes)

    rlb = sub.add_parser("rl-training-batch", help="POST /api/rl/training-batch")
    rlb.add_argument("--symbol", default="BTCUSDT")
    rlb.add_argument("--interval", default="1h")
    rlb.add_argument("--query-length", type=int, default=40)
    rlb.add_argument("--forecast-horizon", type=int, default=24)
    rlb.add_argument("--batch-size", type=int, default=100)
    rlb.add_argument("--min-similarity", type=float, default=0.70)
    rlb.add_argument("--raw", action="store_true")
    rlb.set_defaults(func=cmd_rl_training_batch)

    a = sub.add_parser("audit", help="Backtest + regime attribution for losing trades")
    a.add_argument("--symbol", required=True)
    a.add_argument("--interval", default="1h")
    a.add_argument("--days", type=int, default=90)
    a.add_argument("--q", type=int, default=60)
    a.add_argument("--f", type=int, default=24)
    a.add_argument("--step", type=int, default=24)
    a.add_argument("--top-k", type=int, default=5)
    a.add_argument("--min-prob", type=float, default=0.50)
    a.add_argument(
        "--fee-pct",
        type=float,
        default=None,
        help="Exchange fee in percent (e.g. 0.04)",
    )
    a.add_argument(
        "--slippage-pct",
        type=float,
        default=None,
        help="Execution slippage in percent (e.g. 0.02)",
    )
    a.add_argument("--raw", action="store_true")
    a.set_defaults(func=cmd_audit)

    sc = sub.add_parser(
        "scan",
        help="Watchlist heartbeat: batch-search + regime + stability -> GO/NO-GO",
    )
    sc.add_argument(
        "--symbols",
        required=True,
        help='Comma-separated symbols, e.g. "BTCUSDT,ETHUSDT,SOLUSDT"',
    )
    sc.add_argument("--interval", default="1h")
    sc.add_argument("--q", type=int, default=60)
    sc.add_argument("--f", type=int, default=24)
    sc.add_argument("--limit", type=int, default=16)
    sc.add_argument("--sort", default="similarity")
    sc.add_argument("--force", action="store_true")
    sc.add_argument(
        "--block-regimes",
        default="",
        help='Comma-separated regimes to block, e.g. "BEARISH_MOMENTUM,STABLE_DOWNTREND"',
    )
    sc.add_argument(
        "--stability-offsets",
        default="0,1",
        help='Comma-separated offsets for stability via recalc (default "0,1")',
    )
    sc.add_argument(
        "--stability-threshold",
        type=float,
        default=0.5,
        help="Volatility threshold in percent points for stability (default 0.5)",
    )
    sc.add_argument(
        "--no-stability",
        action="store_true",
        help="Skip stability check (faster; uses only regime + matches)",
    )
    sc.add_argument(
        "--grid-hint",
        action="store_true",
        help="For GO rows only, fetch /api/patterns/grid and add GRID_HINT column",
    )
    sc.add_argument(
        "--grid-hint-human",
        action="store_true",
        help="When used with --grid-hint, print human-friendly labels",
    )
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--watch", action="store_true", help="Repeat scan every N seconds")
    sc.add_argument(
        "--every", type=int, default=60, help="Seconds between scans (default 60)"
    )
    sc.set_defaults(func=cmd_scan)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    common = CommonArgs(base_url=args.base_url, api_key=args.api_key)
    return args.func(common, args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
