#!/usr/bin/env python3
"""
Trading Companion: Automated Pattern-Based Trading Analyst

This script demonstrates a professional trading workflow using the aipp SDK:
1. Discovery: Fetch live signals from the background scanner.
2. Validation: Run a backtest "audit" for each signal to ensure recent performance.
3. Regime Awareness: Check market regime (e.g., Trend vs. Range) and model confidence.
4. Execution: Generate a comprehensive "Trade Card" with entry/exit levels.

Usage:
    export AIPP_BASE_URL="https://aipricepatterns.com/api/rust"
    python trading_companion.py
"""

import os

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize Rich console
console = Console()

# --- SDK Import Helper ---
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Force local SDK import if running from repo
repo_src = (Path(__file__).parent.parent / "src").resolve()
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

try:
    from aipricepatterns import Client
except ImportError:
    console.print(
        "[red]Error: aipricepatterns SDK not found. Please install it or run from the repo root.[/red]"
    )
    sys.exit(1)


def run_audit(client: Client, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    """Runs a quick walk-forward backtest to validate the pattern engine's performance on this asset."""
    try:
        import time

        # Run backtest for the last ~20 days (approx. 500 bars for 1h)
        # Calculate start_ts in ms
        days_ago = 20
        start_ts = int((time.time() - (days_ago * 24 * 60 * 60)) * 1000)

        res = client.backtest(
            symbol=symbol,
            interval=interval,
            q=40,
            f=24,
            step=24,
            start_ts=start_ts,
            include_stats=True,
        )
        return res.get("stats")
    except Exception as e:
        console.print(f"[red]Error auditing {symbol}: {e}[/red]")
        return None


def get_market_context(
    client: Client, symbol: str, interval: str
) -> Optional[Dict[str, Any]]:
    """Retrieves regime and grid-trading guidance."""
    try:
        return client.get_grid_stats(symbol=symbol, interval=interval)
    except Exception as e:
        console.print(f"[red]Error fetching context for {symbol}: {e}[/red]")
        return None


def print_trade_card(
    signal: Dict[str, Any],
    audit_stats: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
):
    """Prints a professional trade card for a signal."""
    symbol = signal.get("symbol", "N/A")
    direction = signal.get("direction", "N/A")
    up_prob = signal.get("up_prob", 0.0)
    interval = signal.get("interval", "1h")

    # 1. Signal Header
    color = "green" if direction == "LONG" else "red"
    header = f"[bold {color}]{direction} SIGNAL: {symbol} ({interval})[/bold {color}]"

    # 2. Backtest Audit Summary
    audit_table = Table(box=box.SIMPLE, show_header=False)
    if audit_stats:
        wr = audit_stats.get("winRate", 0.0)
        tr = audit_stats.get("totalReturnPct", 0.0)
        pf = audit_stats.get("profitFactor", 0.0)
        audit_table.add_row("Recent Win Rate", f"{wr:.1f}%")
        audit_table.add_row("Recent Return", f"{tr:+.2f}%")
        audit_table.add_row("Profit Factor", f"{pf:.2f}")
    else:
        audit_table.add_row(
            "Audit", "[yellow]Skill Audit Failed or Unavailable[/yellow]"
        )

    # 3. Regime & Confidence
    regime_info = "Unknown"
    confidence_info = "N/A"
    if context:
        regime = context.get("regime", {})
        regime_info = regime.get("label", "Unknown")
        conf = context.get("confidence", {})
        confidence_info = f"{conf.get('label', 'N/A')} ({conf.get('score', 0.0):.2f})"

    # 4. Levels & Execution (from grid stats)
    levels_table = Table(box=box.SIMPLE, show_header=False)
    if context:
        rec = context.get("gridRecommendation", {})
        step = rec.get("suggestedStepPct", 0.0)
        levels_table.add_row("Prob. Up", f"{up_prob * 100:.1f}%")
        levels_table.add_row("Regime", f"[cyan]{regime_info}[/cyan]")
        levels_table.add_row("Confidence", confidence_info)
        levels_table.add_row("Volatility (Step)", f"{step:.3f}%")

    # Main Panel Construction
    panel_content = Columns(
        [
            Panel(audit_table, title="[bold]Recent Audit[/bold]", border_style="blue"),
            Panel(
                levels_table,
                title="[bold]Execution Logic[/bold]",
                border_style="magenta",
            ),
        ]
    )

    # Logic Flag: Does the signal align with the regime?
    is_aligned = True
    if regime_info and direction:
        if "DOWNTREND" in regime_info and direction == "LONG":
            is_aligned = False
        elif "UPTREND" in regime_info and direction == "SHORT":
            is_aligned = False

    warning = ""
    if not is_aligned:
        warning = (
            "\n[bold red]⚠️  WARNING: Signal counter-trend to current regime![/bold red]"
        )

    console.print(
        Panel(
            panel_content, title=header, subtitle=warning, expand=False, padding=(1, 2)
        )
    )


def main():
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    console.print(
        Panel(
            f"[bold cyan]AIPP Trading Companion[/bold cyan]\n[dim]Base URL: {base_url}[/dim]",
            border_style="cyan",
        )
    )

    client = Client(base_url=base_url, api_key=api_key)

    console.print("[yellow]🔍 Scanning for high-probability signals...[/yellow]")
    try:
        signals_res = client.get_signals()
        signals = signals_res.get("signals", [])
    except Exception as e:
        console.print(f"[bold red]Critical Error fetching signals: {e}[/bold red]")
        return

    if not signals:
        console.print(
            "[white]No active signals found right now. Check back later.[/white]"
        )
        return

    # Process top 3 signals to keep it concise for the demo
    top_signals = signals[:3]
    console.print(
        f"[green]Found {len(signals)} signals. Analyzing top {len(top_signals)}...[/green]\n"
    )

    for signal in top_signals:
        symbol = signal["symbol"]
        interval = signal["interval"]

        with console.status(f"[bold white]Auditing {symbol}...[/bold white]"):
            audit_stats = run_audit(client, symbol, interval)
            context = get_market_context(client, symbol, interval)

        print_trade_card(signal, audit_stats, context)


if __name__ == "__main__":
    main()
