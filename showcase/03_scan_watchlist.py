from __future__ import annotations

import os
from pathlib import Path

from aipricepatterns.cli import main as aipp_main


def _read_watchlist(path: Path) -> str:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        symbols.append(s)
    return ",".join(symbols)


def main() -> int:
    # Uses env vars AIPP_BASE_URL / AIPP_API_KEY inside the CLI as defaults.
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")

    watchlist_path = Path(__file__).with_name("watchlist.txt")
    symbols = _read_watchlist(watchlist_path)

    # Simple “operations” style scan. Keep stability off for speed.
    argv = [
        "--base-url",
        base_url,
        "scan",
        "--symbols",
        symbols,
        "--interval",
        os.getenv("AIPP_DEMO_INTERVAL", "1h"),
        "--no-stability",
        "--block-regimes",
        os.getenv("AIPP_BLOCK_REGIMES", "BEARISH_MOMENTUM,STABLE_DOWNTREND"),
    ]

    return aipp_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
