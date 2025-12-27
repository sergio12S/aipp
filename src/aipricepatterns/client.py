from typing import Any, Dict, List, Optional

import pandas as pd
import requests


class Client:
    """
    A client for the AI Price Patterns API.

    :param api_key: Your API key (optional for public endpoints).
    :param base_url: The base URL of the API. Defaults to https://aipricepatterns.com/api/rust.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://aipricepatterns.com/api/rust",
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-KEY": api_key})

    def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, json: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=json)
        response.raise_for_status()
        return response.json()

    def search(
        self,
        symbol: str,
        interval: str = "1h",
        q: int = 60,
        f: int = 30,
        top_k: int = 5,
        start: Optional[int] = None,
        window_start_ts: Optional[int] = None,
        window_end_ts: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort: str = "similarity",
        force: bool = False,
        anchor_ts: Optional[int] = None,
        cross_asset: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for historical patterns similar to the current price action.

        :param symbol: Trading pair (e.g., 'BTCUSDT').
        :param interval: Candle interval ('1h', '4h', '1d').
        :param q: Query length (lookback window size).
        :param f: Forecast horizon (how far to predict).
        :param top_k: Number of matches to return.
        :param start: Optional explicit start offset (0 = first bar).
        :param window_start_ts: Restrict search to a candle range (start timestamp in ms).
        :param window_end_ts: Restrict search to a candle range (end timestamp in ms).
        :param filters: Dictionary of filters (e.g., {'minSimilarity': 0.8}).
        :param sort: Sort mode ('similarity', 'recent', 'historic', 'corr', 'rmse').
        :param force: Force-refresh candles from Binance.
        :param anchor_ts: Pin the last bar to a specific timestamp.
        :param cross_asset: If true, search across all indexed symbols for matches.
        :return: Dictionary containing matches and forecast data.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "q": q,
            "f": f,
            "limit": top_k,
            "sort": sort,
            "force": str(force).lower(),
            "crossAsset": str(cross_asset).lower(),
        }
        if start is not None:
            params["start"] = start
        if window_start_ts is not None:
            params["windowStartTs"] = window_start_ts
        if window_end_ts is not None:
            params["windowEndTs"] = window_end_ts
        if filters:
            import json

            params["filters"] = json.dumps(filters)
        if anchor_ts is not None:
            params["anchor_ts"] = anchor_ts

        return self._get("/api/patterns", params=params)

    def get_pattern_metrics(
        self,
        symbol: str,
        interval: str = "1h",
        q: int = 40,
        f: int = 30,
        start: Optional[int] = None,
        window_start_ts: Optional[int] = None,
        window_end_ts: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 32,
    ) -> Dict[str, Any]:
        """
        Get statistical digest for the current query window.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "q": q,
            "f": f,
            "limit": limit,
        }
        if start is not None:
            params["start"] = start
        if window_start_ts is not None:
            params["windowStartTs"] = window_start_ts
        if window_end_ts is not None:
            params["windowEndTs"] = window_end_ts
        if filters:
            import json

            params["filters"] = json.dumps(filters)

        return self._get("/api/patterns/metrics", params=params)

    def get_grid_stats(self, symbol: str, interval: str = "1h") -> Dict[str, Any]:
        """
        Get grid-trading guidance (sigma levels, terminal percentiles, etc.).
        """
        params = {"symbol": symbol, "interval": interval}
        return self._get("/api/patterns/grid", params=params)

    def batch_search(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run multiple searches in a single request.

        :param requests: List of dictionaries, each containing search params (symbol, interval, q, f, etc.)
        """
        return self._post("/api/patterns/batch", json={"requests": requests})

    def get_signals(self) -> Dict[str, Any]:
        """
        Retrieves the latest high-probability signals discovered by the background scanner.
        """
        return self._get("/api/patterns/signals")

    def get_match_details(self, match_id: str) -> Dict[str, Any]:
        """
        Retrieve full detail for a specific match.
        """
        return self._get(f"/api/patterns/matches/{match_id}")

    def search_drawn_pattern(
        self,
        query_values: List[float],
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        forecast_horizon: int = 30,
        limit: int = 32,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Find analogues for an arbitrary sketch or indicator output.
        """
        payload = {
            "symbol": symbol,
            "interval": interval,
            "queryValues": query_values,
            "forecastHorizon": forecast_horizon,
            "limit": limit,
        }
        if filters:
            payload["filters"] = filters
        return self._post("/api/patterns/drawn", json=payload)

    def recalc_patterns(
        self,
        symbol: str,
        interval: str,
        start: int,
        q: int = 60,
        f: int = 30,
        limit: int = 32,
        sort: str = "similarity",
        window_start_ts: Optional[int] = None,
        window_end_ts: Optional[int] = None,
        cursor: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Re-score matches for a different start offset (historical point) without refetching the full series.
        This is used to "scroll" through history and see what patterns existed at that time.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "q": q,
            "f": f,
            "limit": limit,
            "sort": sort,
            "force": str(force).lower(),
        }
        if window_start_ts is not None:
            params["windowStartTs"] = window_start_ts
        if window_end_ts is not None:
            params["windowEndTs"] = window_end_ts
        if cursor is not None:
            params["cursor"] = cursor

        return self._get("/api/patterns-recalc", params=params)

    def backtest(
        self,
        symbol: str,
        interval: str = "1h",
        q: int = 60,
        f: int = 30,
        step: int = 24,
        top_k: int = 5,
        min_prob: float = 0.55,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        include_stats: bool = True,
        fee_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run a walk-forward backtest simulation.

        :param symbol: Trading pair.
        :param interval: Candle interval.
        :param q: Query length.
        :param f: Forecast horizon.
        :param step: Step size in bars (how often to predict).
        :param top_k: Number of neighbors for voting.
        :param min_prob: Minimum probability threshold to take a trade.
        :param include_stats: Calculate equity curve and institutional metrics (Sharpe, Expectancy, etc.).
        :return: Dictionary containing:
            - totalSteps: Total steps in simulation.
            - trades: List of trade details (including neighborOffsets).
            - stats: Dictionary with metrics:
                - expectancy: Average profit per trade.
                - recoveryFactor: Total Return / Max Drawdown.
                - annualizedReturnPct: Projected yearly return.
                - benchmarkReturnPct: Buy & Hold return for the same period.
                - profitFactor: Gross Profit / Gross Loss.
                - sharpeRatio, sortinoRatio, calmarRatio.
                - winStreak, lossStreak.
                - equityCurve: List of {ts, value} points.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "q": q,
            "f": f,
            "step": step,
            "topK": top_k,
            "minProb": min_prob,
            "includeStats": str(include_stats).lower(),
            "feePct": fee_pct,
            "slippagePct": slippage_pct,
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts

        return self._get("/api/patterns/backtest", params=params)

    def backtest_specific_pattern(
        self,
        symbol: str,
        interval: str,
        q: int,
        f: int,
        query_vector: Optional[List[float]] = None,
        timestamp: Optional[int] = None,
        offset: Optional[int] = None,
        top_k: int = 5,
        fee_pct: float = 0.0,
        slippage_pct: float = 0.0,
        include_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze how a SPECIFIC pattern (defined by timestamp or vector) performed historically.

        :param symbol: Trading pair.
        :param timestamp: The timestamp of the pattern to analyze (optional).
        :param query_vector: Explicit vector to search for (optional).
        :return: Analysis of that specific pattern's historical performance.
        """
        payload = {
            "symbol": symbol,
            "interval": interval,
            "q": q,
            "f": f,
            "topK": top_k,
            "includeStats": include_stats,
            "feePct": fee_pct,
            "slippagePct": slippage_pct,
        }
        if timestamp:
            payload["timestamp"] = timestamp
        if offset:
            payload["offset"] = offset
        if query_vector:
            payload["queryVector"] = query_vector

        return self._post("/api/patterns/backtest/specific", json=payload)

    def get_datasets(self) -> pd.DataFrame:
        """
        Get list of available datasets and their status.
        :return: Pandas DataFrame of available symbols and intervals.
        """
        data = self._get("/api/dataset/status")
        # The API returns a list of objects, perfect for DataFrame
        return pd.DataFrame(data)

    def expand_dataset(
        self,
        symbol: str,
        interval: str,
        bars: Optional[int] = None,
        since: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extends stored history backwards by fetching more Binance candles.
        """
        payload = {"symbol": symbol, "interval": interval}
        if bars is not None:
            payload["bars"] = bars
        if since is not None:
            payload["since"] = since
        return self._post("/api/dataset/expand", json=payload)

    def delete_dataset(
        self,
        symbol: str,
        interval: str,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Removes an inclusive time range and rebuilds caches.
        """
        payload = {"symbol": symbol, "interval": interval}
        if from_ts is not None:
            payload["from_ts"] = from_ts
        if to_ts is not None:
            payload["to_ts"] = to_ts
        return self._post("/api/dataset/delete", json=payload)

    def get_dataset_gaps(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Detects missing segments for a (symbol, interval) pair.
        """
        params = {"symbol": symbol, "interval": interval}
        return self._get("/api/dataset/gaps", params=params)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Aggregated statistics across datasets.
        """
        return self._get("/api/dataset/stats")

    def get_dataset_vectors(self) -> Dict[str, Any]:
        """
        Lists all available vector datasets.
        """
        return self._get("/api/dataset/vectors")

    def ann_upsert(
        self,
        id: int,
        vector: List[float],
        symbol: str,
        interval: str,
        payload_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Insert or update a vector in the ANN index.
        """
        payload = {
            "id": id,
            "vector": vector,
            "payload": {"symbol": symbol, "interval": interval},
        }
        if payload_extra:
            payload["payload"].update(payload_extra)
        return self._post("/ann/upsert", json=payload)

    def ann_search(
        self,
        vector: List[float],
        k: int,
        ef: int = 64,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for nearest neighbours directly in the ANN index.
        """
        payload = {"vector": vector, "k": k, "ef": ef}
        if filter:
            payload["filter"] = filter
        return self._post("/ann/search", json=payload)

    def get_ann_status(self) -> Dict[str, Any]:
        """
        Report ANN build progress.
        """
        return self._get("/ann/status")

    def get_dataset_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of loaded datasets (symbols, candle counts, etc.).
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/api/dataset/status", params=params)

    def get_recent_prices(
        self, symbol: str, interval: str = "1h", limit: int = 40
    ) -> List[float]:
        """
        Fetch recent prices for a symbol.
        """
        params = {"symbol": symbol, "interval": interval, "q": limit}
        res = self._get("/api/patterns", params=params)
        series = res.get("series", [])
        return series[-limit:] if len(series) >= limit else series

    def get_rl_simple(
        self,
        symbol: str,
        interval: str = "1h",
        timestamp: Optional[int] = None,
        current_state: Optional[List[float]] = None,
        query_length: int = 40,
        forecast_horizon: int = 24,
        num_episodes: int = 1000,
        min_similarity: float = 0.80,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetch simple RL training episodes (one decision per episode).
        """
        payload = {
            "symbol": symbol,
            "interval": interval,
            "queryLength": query_length,
            "forecastHorizon": forecast_horizon,
            "numEpisodes": num_episodes,
            "minSimilarity": min_similarity,
        }
        if timestamp:
            payload["timestamp"] = timestamp
        if current_state:
            payload["currentState"] = current_state
        if start_ts:
            payload["startTs"] = start_ts
        if end_ts:
            payload["endTs"] = end_ts

        return self._post("/api/rl/simple", json=payload)

    def get_rl_episodes(
        self,
        symbol: str,
        interval: str = "1h",
        current_state: Optional[List[float]] = None,
        query_length: int = 40,
        forecast_horizon: int = 24,
        num_episodes: int = 20,
        min_similarity: float = 0.80,
        include_actions: bool = True,
        reward_type: str = "returns",
        sampling_strategy: str = "diverse",
        anchor_ts: Optional[int] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch full trajectory RL training episodes.
        """
        if current_state is None and anchor_ts is None:
            current_state = self.get_recent_prices(symbol, interval, query_length)

        payload = {
            "symbol": symbol,
            "interval": interval,
            "queryLength": query_length,
            "forecastHorizon": forecast_horizon,
            "numEpisodes": num_episodes,
            "minSimilarity": min_similarity,
            "includeActions": include_actions,
            "rewardType": reward_type,
            "samplingStrategy": sampling_strategy,
        }
        if current_state:
            payload["currentState"] = current_state
        if anchor_ts:
            payload["anchorTs"] = anchor_ts
        if start_ts:
            payload["startTs"] = start_ts
        if end_ts:
            payload["endTs"] = end_ts
        if regime:
            payload["regime"] = regime

        return self._post("/api/rl/episodes", json=payload)

    def get_rl_training_batch(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        query_length: int = 40,
        forecast_horizon: int = 24,
        batch_size: int = 500,
        min_similarity: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Returns flattened arrays optimized for efficient batch training.
        """
        payload = {
            "symbol": symbol,
            "interval": interval,
            "queryLength": query_length,
            "forecastHorizon": forecast_horizon,
            "batchSize": batch_size,
            "minSimilarity": min_similarity,
        }
        return self._post("/api/rl/training-batch", json=payload)

    def get_rl_regimes(
        self, symbol: str = "BTCUSDT", interval: str = "1h"
    ) -> Dict[str, Any]:
        """
        Returns available market regimes for regime-based training strategies.
        """
        params = {"symbol": symbol, "interval": interval}
        return self._get("/api/rl/regimes", params=params)

    def get_current_regime(
        self, symbol: str = "BTCUSDT", interval: str = "1h", query_length: int = 40
    ) -> Dict[str, Any]:
        """
        Detects the current market regime based on the latest price action.
        """
        params = {"symbol": symbol, "interval": interval, "queryLength": query_length}
        return self._get("/api/rl/regimes/latest", params=params)

    def detect_regime(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        query_length: int = 40,
        timestamp: Optional[int] = None,
        current_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Detects the market regime for a specific timestamp or custom state vector.
        """
        payload = {
            "symbol": symbol,
            "interval": interval,
            "queryLength": query_length,
        }
        if timestamp:
            payload["timestamp"] = timestamp
        if current_state:
            payload["currentState"] = current_state

        return self._post("/api/rl/regimes", json=payload)

    # --- Pandas Helpers ---

    def backtest_to_df(self, backtest_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert raw backtest results to a Pandas DataFrame.
        """
        if "trades" not in backtest_result:
            return pd.DataFrame()

        df = pd.DataFrame(backtest_result["trades"])
        if not df.empty and "ts" in df.columns:
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("datetime", inplace=True)
        return df

    def equity_curve_to_df(self, backtest_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract the equity curve as a Pandas DataFrame.
        """
        if "stats" not in backtest_result or not backtest_result["stats"]:
            return pd.DataFrame()

        curve = backtest_result["stats"].get("equityCurve", [])
        df = pd.DataFrame(curve)
        if not df.empty and "ts" in df.columns:
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("datetime", inplace=True)
        return df
