from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from aipricepatterns import Client


def _period_anchor_ms(days_ago: int) -> int:
    return int((time.time() - days_ago * 24 * 60 * 60) * 1000)


def _require_rl_deps() -> Tuple[Any, Any, Any, Any]:
    """Import RL deps lazily so the showcase runs even without extra packages."""

    try:
        import numpy as np  # type: ignore
        import gymnasium as gym  # type: ignore
        from gymnasium import spaces  # type: ignore
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

        return np, gym, spaces, (PPO, DummyVecEnv)
    except Exception as e:
        print("Missing RL dependencies for this demo.")
        print("Install:")
        print('  pip install "aipricepatterns[rl]"')
        print("Error:")
        print(f"  {e}")
        raise SystemExit(2)


def _safe_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return default


def _dd_magnitude(x: Any) -> float:
    """Convert various drawdown encodings into a positive magnitude."""

    v = _safe_float(x, 0.0)
    if v < 0:
        return -v
    return abs(v)


def _map_suggested_action_to_discrete(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        # Some APIs encode: -1/0/1 or 0/1/2; keep conservative.
        v = int(x)
        if v in (-1, 0, 1):
            return 2 if v == -1 else (1 if v == 1 else 0)
        if v in (0, 1, 2):
            return v
        return 0
    if not isinstance(x, str):
        return 0

    s = x.strip().lower()
    if s in ("hold", "flat", "none", "neutral", "wait"):
        return 0
    if s in ("long", "buy", "bull", "up"):
        return 1
    if s in ("short", "sell", "bear", "down"):
        return 2
    return 0


def _simulate_episode(
    np: Any,
    episode: Dict[str, Any],
    *,
    horizon: int,
    trade_cost_pct: float,
    dd_penalty: float,
    policy_fn,
) -> Dict[str, Any]:
    transitions = episode.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return {
            "steps": 0,
            "totalReward": 0.0,
            "totalPnl": 0.0,
            "totalCost": 0.0,
            "totalDdPenalty": 0.0,
            "trades": 0,
            "actionCounts": {0: 0, 1: 0, 2: 0},
            "finalPos": 0,
            "maxDdMag": 0.0,
        }

    episode_similarity = _safe_float(episode.get("similarity"), 0.0)
    base_price = _safe_float((transitions[0] or {}).get("price", 1.0), 1.0) or 1.0

    pos = 0
    dd_prev = 0.0

    total_reward = 0.0
    total_pnl = 0.0
    total_cost = 0.0
    total_dd_pen = 0.0
    trades = 0
    action_counts = {0: 0, 1: 0, 2: 0}
    max_dd_mag = 0.0

    steps = min(horizon, len(transitions))
    for step_idx in range(steps):
        prev = transitions[step_idx - 1] if step_idx > 0 else None
        if isinstance(prev, dict):
            price = _safe_float(prev.get("price", 0.0), 0.0)
            ret_prev = _safe_float(prev.get("ret", 0.0), 0.0)
            vol = _safe_float(prev.get("volatility", 0.0), 0.0)
            cum_ret = _safe_float(prev.get("cumulativeReturn", 0.0), 0.0)
            dd_mag_obs = _dd_magnitude(prev.get("maxDrawdown", 0.0))
            price_rel = (price / (base_price if base_price != 0 else 1.0)) - 1.0
        else:
            price_rel = 0.0
            ret_prev = 0.0
            vol = 0.0
            cum_ret = 0.0
            dd_mag_obs = 0.0

        time_left = 1.0 - (step_idx / float(max(1, horizon)))
        obs = np.array(
            [
                price_rel,
                ret_prev,
                vol,
                cum_ret,
                dd_mag_obs,
                float(pos),
                time_left,
                float(episode_similarity),
            ],
            dtype=np.float32,
        )

        action = int(policy_fn(obs, step_idx))
        action = 0 if action not in (0, 1, 2) else action
        action_counts[action] = action_counts.get(action, 0) + 1

        t = transitions[step_idx]
        t = t if isinstance(t, dict) else {}
        ret = _safe_float(t.get("ret", 0.0), 0.0)

        old_pos = int(pos)
        if action == 1:
            new_pos = 1
        elif action == 2:
            new_pos = -1
        else:
            new_pos = old_pos

        trade_cost = 0.0
        if new_pos != old_pos and trade_cost_pct != 0.0:
            trade_cost = abs(new_pos - old_pos) * (trade_cost_pct / 100.0)
            trades += 1

        pos = new_pos

        dd_mag = _dd_magnitude(t.get("maxDrawdown", 0.0))
        dd_mag = max(dd_mag, dd_prev)
        dd_increase = max(0.0, dd_mag - dd_prev)
        dd_pen = dd_penalty * dd_increase
        dd_prev = dd_mag
        max_dd_mag = max(max_dd_mag, dd_mag)

        pnl = float(pos) * ret
        reward = pnl - trade_cost - dd_pen

        total_pnl += pnl
        total_cost += trade_cost
        total_dd_pen += dd_pen
        total_reward += reward

    return {
        "steps": steps,
        "totalReward": float(total_reward),
        "totalPnl": float(total_pnl),
        "totalCost": float(total_cost),
        "totalDdPenalty": float(total_dd_pen),
        "trades": int(trades),
        "actionCounts": action_counts,
        "finalPos": int(pos),
        "maxDdMag": float(max_dd_mag),
    }


def _make_env_class():
    np, gym, spaces, _ = _require_rl_deps()

    class ContextAwareTradingEnv(gym.Env):  # type: ignore[misc]
        """Minimal context-aware trading env (Gymnasium-compatible)."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            episodes: List[Dict[str, Any]],
            *,
            max_steps: int,
            trade_cost_pct: float,
            dd_penalty: float,
        ):
            super().__init__()
            self._np = np
            self.episodes = episodes
            self.max_steps = max_steps

            self.trade_cost_pct = float(trade_cost_pct)
            self.dd_penalty = float(dd_penalty)

            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            )

            self._transitions: List[Dict[str, Any]] = []
            self._step_idx = 0
            self._base_price = 1.0
            self._episode_similarity = 0.0
            self._position = 0  # -1 short, 0 flat, +1 long
            self._dd_mag_prev = 0.0

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ):
            super().reset(seed=seed)
            if seed is not None:
                self._np.random.seed(seed)

            episode = self.episodes[int(self._np.random.randint(0, len(self.episodes)))]
            self._episode_similarity = _safe_float(episode.get("similarity"), 0.0)
            transitions = episode.get("transitions") or []
            if not isinstance(transitions, list) or not transitions:
                transitions = []

            self._transitions = [t for t in transitions if isinstance(t, dict)]
            self._step_idx = 0
            self._position = 0
            self._dd_mag_prev = 0.0
            self._base_price = float(
                (self._transitions[0].get("price") if self._transitions else 1.0) or 1.0
            )

            return self._get_obs(), {}

        def step(self, action: int):
            if (
                self._step_idx >= len(self._transitions)
                or self._step_idx >= self.max_steps
            ):
                return self._get_obs(), 0.0, True, False, {}

            t = self._transitions[self._step_idx]
            ret = _safe_float(t.get("ret", 0.0), 0.0)

            # Action semantics: 0=hold, 1=set long, 2=set short.
            old_pos = int(self._position)
            if action == 1:
                new_pos = 1
            elif action == 2:
                new_pos = -1
            else:
                new_pos = old_pos

            # Transaction cost (pct-per-trade). Uses the same semantics as feePct elsewhere:
            # 0.04 means 0.04% per trade.
            trade_cost = 0.0
            if new_pos != old_pos and self.trade_cost_pct != 0.0:
                trade_cost = abs(new_pos - old_pos) * (self.trade_cost_pct / 100.0)

            self._position = new_pos

            # Reward = position * return - trade_cost - drawdown_increase_penalty.
            dd_mag = _dd_magnitude(t.get("maxDrawdown", 0.0))
            dd_mag = max(dd_mag, self._dd_mag_prev)
            dd_increase = max(0.0, dd_mag - self._dd_mag_prev)
            dd_pen = self.dd_penalty * dd_increase
            self._dd_mag_prev = dd_mag

            pnl = float(self._position) * ret
            reward = pnl - trade_cost - dd_pen

            self._step_idx += 1
            terminated = self._step_idx >= min(len(self._transitions), self.max_steps)

            info = {
                "ret": ret,
                "position": int(self._position),
                "tradeCost": float(trade_cost),
                "ddMag": float(dd_mag),
                "ddPenalty": float(dd_pen),
                "pnl": float(pnl),
                "suggestedAction": t.get("suggestedAction"),
            }
            return self._get_obs(), float(reward), terminated, False, info

        def _get_obs(self):
            if not self._transitions or self._step_idx <= 0:
                price_rel = 0.0
                ret_prev = 0.0
                vol = 0.0
                cum_ret = 0.0
                dd_mag = 0.0
            else:
                prev = self._transitions[self._step_idx - 1]
                price = _safe_float(prev.get("price", 0.0), 0.0)
                ret_prev = _safe_float(prev.get("ret", 0.0), 0.0)
                vol = _safe_float(prev.get("volatility", 0.0), 0.0)
                cum_ret = _safe_float(prev.get("cumulativeReturn", 0.0), 0.0)
                dd_mag = _dd_magnitude(prev.get("maxDrawdown", 0.0))
                base = self._base_price if self._base_price != 0 else 1.0
                price_rel = (price / base) - 1.0

            time_left = 1.0 - (self._step_idx / float(max(1, self.max_steps)))
            return self._np.array(
                [
                    price_rel,
                    ret_prev,
                    vol,
                    cum_ret,
                    dd_mag,
                    float(self._position),
                    time_left,
                    float(self._episode_similarity),
                ],
                dtype=self._np.float32,
            )

    return ContextAwareTradingEnv


def main() -> int:
    np, gym, _, (PPO, DummyVecEnv) = _require_rl_deps()
    EnvCls = _make_env_class()

    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    api_key = os.getenv("AIPP_API_KEY")

    symbol = os.getenv("AIPP_RL_SYMBOL", "BTCUSDT")
    interval = os.getenv("AIPP_RL_INTERVAL", "1h")

    # Use a meaningful moment (e.g., SVB-ish crisis) if you have it.
    anchor_ts = int(os.getenv("AIPP_RL_ANCHOR_TS", str(_period_anchor_ms(30))))

    forecast_horizon = int(os.getenv("AIPP_RL_HORIZON", "24"))
    num_episodes = int(os.getenv("AIPP_RL_NUM_EPISODES", "50"))
    min_similarity = float(os.getenv("AIPP_RL_MIN_SIMILARITY", "0.80"))

    # More realistic shaping knobs (kept explicit and easy to tweak).
    # trade_cost_pct uses the same semantics as feePct elsewhere: 0.04 means 0.04%.
    trade_cost_pct = float(os.getenv("AIPP_RL_TRADE_COST_PCT", "0.00"))
    # dd_penalty penalizes increases in drawdown magnitude.
    dd_penalty = float(os.getenv("AIPP_RL_DD_PENALTY", "0.10"))

    # Baseline gating: follow suggestedAction only when episode similarity is high enough.
    suggested_min_similarity = float(
        os.getenv("AIPP_RL_SUGGESTED_MIN_SIMILARITY", "0.90")
    )

    sanity_episodes = int(os.getenv("AIPP_RL_SANITY_EPISODES", "10"))

    # Optional: fetch a broader similarity spectrum for sanity evaluation only.
    # This does NOT affect training data unless you also lower AIPP_RL_MIN_SIMILARITY.
    sanity_min_similarity_env = os.getenv("AIPP_RL_SANITY_MIN_SIMILARITY")
    sanity_min_similarity_override = (
        float(sanity_min_similarity_env)
        if isinstance(sanity_min_similarity_env, str)
        and sanity_min_similarity_env.strip()
        else None
    )
    sanity_fetch_episodes = int(
        os.getenv(
            "AIPP_RL_SANITY_FETCH_EPISODES",
            str(max(num_episodes, sanity_episodes * 8, 80)),
        )
    )

    total_timesteps = int(os.getenv("AIPP_RL_TRAIN_TIMESTEPS", "5000"))
    learning_rate = float(os.getenv("AIPP_RL_LR", "0.001"))

    client = Client(base_url=base_url, api_key=api_key)

    print("=" * 78)
    print("RL — TRAIN A SPECIALIST PPO AGENT (CONTEXT-AWARE)")
    print("=" * 78)
    print(f"Base URL: {base_url}")
    print(
        f"Symbol: {symbol}  Interval: {interval}  anchorTs={anchor_ts}  horizon={forecast_horizon}"
    )
    print(
        f"Episodes: {num_episodes}  minSimilarity={min_similarity:.2f}  timesteps={total_timesteps}"
    )
    print(f"Shaping: tradeCostPct={trade_cost_pct:.4f}%  ddPenalty={dd_penalty:.3f}")
    print(f"Baselines: suggestedMinSimilarity={suggested_min_similarity:.2f}")
    print(f"Sanity: episodes={sanity_episodes}")

    res = client.get_rl_episodes(
        symbol=symbol,
        interval=interval,
        anchor_ts=anchor_ts,
        forecast_horizon=forecast_horizon,
        num_episodes=num_episodes,
        min_similarity=min_similarity,
        include_actions=True,
        reward_type="returns",
        sampling_strategy="uniform",
    )

    episodes = res.get("episodes") if isinstance(res, dict) else None
    if not isinstance(episodes, list) or not episodes:
        print("No episodes returned.")
        print(
            "Try lowering AIPP_RL_MIN_SIMILARITY (e.g. 0.70) or changing AIPP_RL_ANCHOR_TS."
        )
        return 1

    # Wrap as a Gymnasium env for SB3.
    def make_env():
        return EnvCls(
            episodes,
            max_steps=forecast_horizon,
            trade_cost_pct=trade_cost_pct,
            dd_penalty=dd_penalty,
        )

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=learning_rate,
    )

    print("-")
    print("Training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training done.")

    # ---------------------------------------------------------------------
    # Sanity baselines
    # Compare policies on both a high-sim and low-sim episode to show why
    # confidence gating matters.
    # ---------------------------------------------------------------------
    dict_episodes = [e for e in episodes if isinstance(e, dict)]
    high_sim_episode = max(
        dict_episodes, key=lambda e: _safe_float(e.get("similarity"), 0.0)
    )
    low_sim_episode = min(
        dict_episodes, key=lambda e: _safe_float(e.get("similarity"), 0.0)
    )

    def ppo_policy(obs, _step_idx: int) -> int:
        a, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        return int(a[0])

    def always_flat(_obs, _step_idx: int) -> int:
        return 0

    def always_long(_obs, _step_idx: int) -> int:
        return 1

    def _pick_sanity_episodes(
        sorted_eps: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        if k >= len(sorted_eps):
            return list(sorted_eps)
        # Evenly sample across similarity spectrum.
        idxs = np.linspace(0, len(sorted_eps) - 1, num=k, dtype=int)
        out: List[Dict[str, Any]] = []
        seen = set()
        for i in idxs.tolist():
            if i in seen:
                continue
            seen.add(i)
            out.append(sorted_eps[int(i)])
        return out

    def _summarize_over_episodes(
        label: str,
        eps: List[Dict[str, Any]],
        policy_builder,
        *,
        gate_rate: Optional[float] = None,
    ) -> None:
        if not eps:
            return

        rewards: List[float] = []
        pnls: List[float] = []
        costs: List[float] = []
        dd_pens: List[float] = []
        trades: List[int] = []
        max_dds: List[float] = []

        for ep in eps:
            r = _simulate_episode(
                np,
                ep,
                horizon=forecast_horizon,
                trade_cost_pct=trade_cost_pct,
                dd_penalty=dd_penalty,
                policy_fn=policy_builder(ep),
            )
            rewards.append(float(r["totalReward"]))
            pnls.append(float(r["totalPnl"]))
            costs.append(float(r["totalCost"]))
            dd_pens.append(float(r["totalDdPenalty"]))
            trades.append(int(r["trades"]))
            max_dds.append(float(r["maxDdMag"]))

        reward_avg = float(np.mean(rewards))
        reward_med = float(np.median(rewards))
        pnl_avg = float(np.mean(pnls))
        cost_avg = float(np.mean(costs))
        ddpen_avg = float(np.mean(dd_pens))
        trade_avg = float(np.mean(trades))
        maxdd_avg = float(np.mean(max_dds))
        extra = (
            f"  gateON={gate_rate * 100:.0f}%" if isinstance(gate_rate, float) else ""
        )

        print(
            f"  {label:14} avgReward={reward_avg:+.4f}  medReward={reward_med:+.4f}  avgPnl={pnl_avg:+.4f}  avgCost={cost_avg:+.4f}"
        )
        print(
            f"  {'':14} avgDdPen={ddpen_avg:+.4f}  avgTrades={trade_avg:.2f}  avgMaxDD={maxdd_avg:.4f}{extra}"
        )

    # Aggregate sanity check over K episodes (sampled across similarity spectrum)
    def _episode_key(ep: Dict[str, Any]) -> str:
        ep_id = ep.get("id")
        if isinstance(ep_id, (str, int)):
            return f"id:{ep_id}"
        ts = ep.get("transitions")
        first = ts[0] if isinstance(ts, list) and ts and isinstance(ts[0], dict) else {}
        first_ts = first.get("ts") or first.get("t") or first.get("timestamp")
        first_price = first.get("price")
        return (
            f"sim:{_safe_float(ep.get('similarity'), 0.0):.6f}|"
            f"n:{len(ts) if isinstance(ts, list) else 0}|"
            f"t0:{first_ts}|p0:{first_price}"
        )

    def _merge_unique_episodes(
        base: List[Dict[str, Any]], extra: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for ep in base + extra:
            if not isinstance(ep, dict):
                continue
            k = _episode_key(ep)
            if k in seen:
                continue
            seen.add(k)
            out.append(ep)
        return out

    sanity_source = "train"
    dict_episodes = [e for e in episodes if isinstance(e, dict)]
    dict_episodes_all = list(dict_episodes)

    def _sorted_by_similarity(eps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(eps, key=lambda e: _safe_float(e.get("similarity"), 0.0))

    dict_episodes_sorted = _sorted_by_similarity(dict_episodes_all)
    sanity_eps = _pick_sanity_episodes(dict_episodes_sorted, sanity_episodes)

    def build_suggested_action(ep: Dict[str, Any]):
        def suggested_action(_obs, step_idx: int) -> int:
            ts = ep.get("transitions")
            if not isinstance(ts, list) or step_idx >= len(ts):
                return 0
            t = ts[step_idx] if isinstance(ts[step_idx], dict) else {}
            return _map_suggested_action_to_discrete(t.get("suggestedAction"))

        return suggested_action

    def build_suggested_if_conf(ep: Dict[str, Any]):
        sim = _safe_float(ep.get("similarity"), 0.0)
        conf_ok = sim >= suggested_min_similarity

        suggested_action = build_suggested_action(ep)

        def suggested_if_confident(_obs, step_idx: int) -> int:
            if not conf_ok:
                return 0
            return suggested_action(_obs, step_idx)

        return suggested_if_confident

    def build_always_flat(_ep: Dict[str, Any]):
        return always_flat

    def build_always_long(_ep: Dict[str, Any]):
        return always_long

    def build_ppo(_ep: Dict[str, Any]):
        return ppo_policy

    if sanity_eps:
        gate_on = sum(
            1
            for ep in sanity_eps
            if _safe_float(ep.get("similarity"), 0.0) >= suggested_min_similarity
        )
        gate_rate = gate_on / float(len(sanity_eps))
        sims = [_safe_float(ep.get("similarity"), 0.0) for ep in sanity_eps]

        # If sanity sample is too "high-sim" (gate always ON/OFF or tiny sim range),
        # refetch a wider pool for sanity evaluation only.
        sim_range = (max(sims) - min(sims)) if sims else 0.0
        need_more_variety = (
            (sim_range < 0.05) or (gate_rate >= 0.99) or (gate_rate <= 0.01)
        )
        if need_more_variety and sanity_fetch_episodes > 0:
            inferred_min_sim = max(
                0.0,
                min(min_similarity, suggested_min_similarity) - 0.20,
            )
            sanity_query_min_similarity = (
                float(sanity_min_similarity_override)
                if isinstance(sanity_min_similarity_override, float)
                else float(inferred_min_sim)
            )
            if sanity_query_min_similarity < min_similarity:
                try:
                    res_sanity = client.get_rl_episodes(
                        symbol=symbol,
                        interval=interval,
                        anchor_ts=anchor_ts,
                        forecast_horizon=forecast_horizon,
                        num_episodes=sanity_fetch_episodes,
                        min_similarity=sanity_query_min_similarity,
                        include_actions=True,
                        reward_type="returns",
                        sampling_strategy="uniform",
                    )
                    sanity_eps_raw = (
                        res_sanity.get("episodes")
                        if isinstance(res_sanity, dict)
                        else None
                    )
                    if isinstance(sanity_eps_raw, list) and sanity_eps_raw:
                        sanity_source = f"sanity(minSim={sanity_query_min_similarity:.2f}, n={sanity_fetch_episodes})"
                        dict_episodes_all = _merge_unique_episodes(
                            dict_episodes_all,
                            [e for e in sanity_eps_raw if isinstance(e, dict)],
                        )
                        dict_episodes_sorted = _sorted_by_similarity(dict_episodes_all)
                        sanity_eps = _pick_sanity_episodes(
                            dict_episodes_sorted, sanity_episodes
                        )
                        gate_on = sum(
                            1
                            for ep in sanity_eps
                            if _safe_float(ep.get("similarity"), 0.0)
                            >= suggested_min_similarity
                        )
                        gate_rate = gate_on / float(len(sanity_eps))
                        sims = [
                            _safe_float(ep.get("similarity"), 0.0) for ep in sanity_eps
                        ]
                        sim_range = (max(sims) - min(sims)) if sims else 0.0
                except Exception:
                    # If sanity refetch fails (server/network), keep original sample.
                    sanity_source = "train"

        print("-")
        print(
            f"Sanity baselines summary (K={len(sanity_eps)}, sim range {min(sims):.3f}..{max(sims):.3f}, gate={suggested_min_similarity:.2f}, source={sanity_source}):"
        )
        _summarize_over_episodes("PPO", sanity_eps, build_ppo)
        _summarize_over_episodes("suggested", sanity_eps, build_suggested_action)
        _summarize_over_episodes(
            "suggestedIfConf", sanity_eps, build_suggested_if_conf, gate_rate=gate_rate
        )
        _summarize_over_episodes("always long", sanity_eps, build_always_long)
        _summarize_over_episodes("always flat", sanity_eps, build_always_flat)

        print("-")
        print("Investor read:")
        print(
            "  Expect PPO to beat always-flat and (ideally) suggestedIfConf net of costs and drawdown penalties."
        )
        print(
            "  If PPO only matches/loses to trivial baselines, treat it as non-learning (no edge) for this context."
        )
        if sim_range < 0.05 or gate_rate >= 0.99 or gate_rate <= 0.01:
            print(
                "  Note: gating signal may be saturated; try raising AIPP_RL_SUGGESTED_MIN_SIMILARITY or lowering AIPP_RL_MIN_SIMILARITY."
            )

    # ---------------------------------------------------------------------
    # Sanity baselines (single-episode detail)
    # Compare policies on both a high-sim and low-sim episode.
    # ---------------------------------------------------------------------
    def _print_baselines_for_episode(ep: Dict[str, Any], label: str) -> None:
        sim = _safe_float(ep.get("similarity"), 0.0)
        conf_ok = sim >= suggested_min_similarity

        def suggested_action(_obs, step_idx: int) -> int:
            ts = ep.get("transitions")
            if not isinstance(ts, list) or step_idx >= len(ts):
                return 0
            t = ts[step_idx] if isinstance(ts[step_idx], dict) else {}
            return _map_suggested_action_to_discrete(t.get("suggestedAction"))

        def suggested_if_confident(_obs, step_idx: int) -> int:
            if not conf_ok:
                return 0
            return suggested_action(_obs, step_idx)

        print("-")
        print(
            f"Sanity baselines ({label}, similarity={sim:.3f}, gate={suggested_min_similarity:.2f} -> {'ON' if conf_ok else 'OFF'}):"
        )

        results = []
        for name, fn in (
            ("PPO", ppo_policy),
            ("suggestedAction", suggested_action),
            ("suggestedIfConf", suggested_if_confident),
            ("always long", always_long),
            ("always flat", always_flat),
        ):
            r = _simulate_episode(
                np,
                ep,
                horizon=forecast_horizon,
                trade_cost_pct=trade_cost_pct,
                dd_penalty=dd_penalty,
                policy_fn=fn,
            )
            results.append((name, r))

        for name, r in results:
            ac = r["actionCounts"]
            print(
                f"  {name:14} reward={r['totalReward']:+.4f}  pnl={r['totalPnl']:+.4f}  cost={r['totalCost']:+.4f}  ddPen={r['totalDdPenalty']:+.4f}"
            )
            print(
                f"{'':16} trades={r['trades']}  maxDD={r['maxDdMag']:.4f}  actions: hold={ac.get(0, 0)} long={ac.get(1, 0)} short={ac.get(2, 0)}"
            )

    _print_baselines_for_episode(high_sim_episode, "high-sim")
    _print_baselines_for_episode(low_sim_episode, "low-sim")

    # Quick evaluation roll-out
    obs = vec_env.reset()
    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}

    for _ in range(forecast_horizon):
        action, _ = model.predict(obs, deterministic=True)
        a = int(action[0])
        action_counts[a] = action_counts.get(a, 0) + 1
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
        if bool(done[0]):
            break

    print("-")
    print("One rollout (deterministic):")
    print(f"  totalReward: {total_reward:+.4f}")
    print(
        f"  actions: hold={action_counts[0]}  long={action_counts[1]}  short={action_counts[2]}"
    )

    print("-")
    print("Takeaway:")
    print(
        "  This demonstrates the core RL workflow: retrieve episodes -> train specialist -> act."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
