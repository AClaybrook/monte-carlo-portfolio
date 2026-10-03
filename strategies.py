"""
Dynamic allocation strategies.

A strategy maps a MarketContext to target weights, shape (paths, assets).
The engine decides what those weights are applied to (see StrategyConfig.apply_to):
- 'contributions': only new cash is split by the strategy's weights
- 'rebalance': holdings are traded to the strategy's weights whenever the
  target changes (checked at StrategyConfig.check_frequency) and on calendar
  rebalance dates
- 'both'

All indicators are computed from PRICES, never from holdings, so contributions
and trades cannot fake a drawdown or a recovery. Strategies are vectorized over
paths: the same code runs on one historical path or 10,000 simulated ones.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class MarketContext:
    """Everything a strategy may look at on a decision day. Arrays are (paths, assets)."""
    current_holdings: np.ndarray          # Dollar value per asset
    current_drawdowns: np.ndarray         # Price drawdown from each asset's running peak (<= 0)
    base_allocations: np.ndarray          # Configured target weights (assets,)
    asset_tickers: List[str]
    current_day: int
    total_days: int

    # Rolling statistics over the engine's indicator window (None until warm)
    rolling_returns: Optional[np.ndarray] = None      # Annualized mean return
    rolling_volatility: Optional[np.ndarray] = None   # Annualized vol
    rolling_sharpe: Optional[np.ndarray] = None
    momentum_score: Optional[np.ndarray] = None       # Compounded return over the window

    portfolio_drawdown: Optional[np.ndarray] = None   # (paths,) drawdown of the time-weighted index
    # (paths,) drawdown of the base-weight portfolio (daily rebalanced). Unlike
    # portfolio_drawdown it keeps moving while a strategy is out of the market.
    reference_drawdown: Optional[np.ndarray] = None
    portfolio_volatility: Optional[np.ndarray] = None # (paths,) annualized, over the window
    current_weights: Optional[np.ndarray] = None
    current_date: Optional[object] = None             # Historical runs only
    periods_per_year: float = 252.0
    market_regime: Optional[np.ndarray] = None
    price_history: Optional[np.ndarray] = None

    # Engine-provided accessors for strategy-specific lookbacks
    _trailing_asset_returns: Optional[Callable[[int], Optional[np.ndarray]]] = None
    _trailing_portfolio_returns: Optional[Callable[[int], Optional[np.ndarray]]] = None

    @property
    def n_paths(self) -> int:
        return self.current_holdings.shape[0]

    def base(self) -> np.ndarray:
        return np.tile(self.base_allocations, (self.n_paths, 1)).astype(float)

    def ticker_index(self, ticker: str) -> Optional[int]:
        upper = [t.upper() for t in self.asset_tickers]
        return upper.index(ticker.upper()) if ticker and ticker.upper() in upper else None

    def trailing_return(self, lookback: int) -> Optional[np.ndarray]:
        """Compounded return per asset over the last `lookback` days, or None if not warm."""
        if self._trailing_asset_returns is not None:
            r = self._trailing_asset_returns(lookback)
            if r is not None:
                return np.prod(1 + r, axis=1) - 1
            return None
        return self.momentum_score

    def trailing_portfolio_volatility(self, lookback: int) -> Optional[np.ndarray]:
        if self._trailing_portfolio_returns is not None:
            r = self._trailing_portfolio_returns(lookback)
            if r is not None:
                return np.std(r, axis=1, ddof=1) * np.sqrt(self.periods_per_year)
            return None
        return self.portfolio_volatility


class AllocationStrategy(ABC):
    # The engine only computes MarketContext.rolling_* / momentum_score /
    # portfolio_volatility when this is True (they are costly on daily checks).
    # Built-in strategies use the trailing_* accessors instead.
    uses_rolling_stats = True

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.lookback_days = 0  # Longest history window the strategy reads

    def reset(self, n_paths: int, asset_tickers: List[str]):
        """Called at the start of every run; strategies with state clear it here."""

    @abstractmethod
    def get_allocation(self, context: MarketContext) -> np.ndarray:
        """Return target weights, shape (paths, assets), rows summing to 1."""

    def get_config_summary(self) -> Dict:
        return {"name": self.name}


def _scale_others(weights: np.ndarray, mask: np.ndarray, idx: int, target_weight: float,
                  base: np.ndarray):
    """Set asset idx to target_weight on masked rows, scaling the rest to fill 1 - target."""
    others = base.sum() - base[idx]
    weights[mask] = 0.0
    weights[mask, idx] = target_weight
    if others > 0:
        for i in range(len(base)):
            if i != idx:
                weights[mask, i] = base[i] / others * (1.0 - target_weight)
    else:
        weights[mask, idx] = 1.0


class StaticAllocationStrategy(AllocationStrategy):
    """Always the configured weights."""
    uses_rolling_stats = False

    def __init__(self):
        super().__init__(name="Static")

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        return context.base()


class BuyTheDipStrategy(AllocationStrategy):
    """Put `aggressive_weight` into target_ticker while its price is more than
    `threshold` below its running peak; otherwise use base weights."""
    uses_rolling_stats = False

    def __init__(self, target_ticker: str, threshold: float = 0.10,
                 aggressive_weight: float = 0.80):
        super().__init__(name=f"Buy the Dip ({target_ticker})")
        self.target_ticker = target_ticker.upper()
        self.threshold = threshold
        self.aggressive_weight = aggressive_weight

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        weights = context.base()
        idx = context.ticker_index(self.target_ticker)
        if idx is None:
            return weights
        dip = context.current_drawdowns[:, idx] < -self.threshold
        if np.any(dip):
            _scale_others(weights, dip, idx, self.aggressive_weight, context.base_allocations)
        return weights

    def get_config_summary(self) -> Dict:
        return {"name": self.name, "target": self.target_ticker,
                "threshold": f"{self.threshold:.0%}",
                "aggressive_weight": f"{self.aggressive_weight:.0%}"}


class CryptoOpportunisticStrategy(AllocationStrategy):
    """Hold `normal_weight` of a crypto asset (base weight if None), raising it
    to `dip_weight` while the crypto price is more than `dip_threshold` below its peak."""
    uses_rolling_stats = False

    def __init__(self, crypto_ticker: str = "BTC-USD", dip_threshold: float = 0.25,
                 normal_weight: Optional[float] = None, dip_weight: float = 0.40):
        super().__init__(name=f"Crypto Opportunistic ({crypto_ticker})")
        self.crypto = crypto_ticker.upper()
        self.threshold = dip_threshold
        self.normal_weight = normal_weight
        self.dip_weight = dip_weight

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        weights = context.base()
        idx = context.ticker_index(self.crypto)
        if idx is None:
            return weights
        dip = context.current_drawdowns[:, idx] < -self.threshold
        if self.normal_weight is not None:
            _scale_others(weights, ~dip, idx, self.normal_weight, context.base_allocations)
        if np.any(dip):
            _scale_others(weights, dip, idx, self.dip_weight, context.base_allocations)
        return weights

    def get_config_summary(self) -> Dict:
        return {"name": self.name, "dip_threshold": f"{self.threshold:.0%}",
                "normal_weight": "base" if self.normal_weight is None else f"{self.normal_weight:.0%}",
                "dip_weight": f"{self.dip_weight:.0%}"}


class MomentumStrategy(AllocationStrategy):
    """Blend base weights with weights proportional to (shifted) trailing return."""
    uses_rolling_stats = False

    def __init__(self, momentum_lookback: int = 63, tilt_strength: float = 0.5,
                 min_weight: float = 0.05):
        super().__init__(name="Momentum Tilt")
        self.lookback = momentum_lookback
        self.tilt_strength = tilt_strength
        self.min_weight = min_weight
        self.lookback_days = momentum_lookback

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        base = context.base()
        mom = context.trailing_return(self.lookback)
        if mom is None:
            return base
        shifted = mom - mom.min(axis=1, keepdims=True) + 1e-6
        mom_weights = shifted / shifted.sum(axis=1, keepdims=True)
        blended = (1 - self.tilt_strength) * base + self.tilt_strength * mom_weights
        blended = np.maximum(blended, self.min_weight)
        return blended / blended.sum(axis=1, keepdims=True)


class VolatilityTargetStrategy(AllocationStrategy):
    """Scale risky weights by target_vol / realized portfolio vol (correlations
    included, since it is measured on the portfolio's own returns). The freed or
    needed weight goes to safe_ticker; without one, weights are renormalized."""
    uses_rolling_stats = False

    def __init__(self, target_vol: float = 0.15, vol_lookback: int = 21,
                 equity_tickers: List[str] = None, safe_ticker: str = None,
                 min_scale: float = 0.25, max_scale: float = 1.5):
        super().__init__(name="Volatility Target")
        self.target_vol = target_vol
        self.lookback = vol_lookback
        self.equity_tickers = [t.upper() for t in (equity_tickers or [])]
        self.safe_ticker = safe_ticker.upper() if safe_ticker else None
        self.min_scale, self.max_scale = min_scale, max_scale
        self.lookback_days = vol_lookback

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        weights = context.base()
        vol = context.trailing_portfolio_volatility(self.lookback)
        if vol is None:
            return weights
        scale = np.clip(self.target_vol / np.maximum(vol, 1e-6), self.min_scale, self.max_scale)
        tickers = [t.upper() for t in context.asset_tickers]
        risky = np.array([t in self.equity_tickers for t in tickers])
        if not risky.any():
            risky = np.array([t != self.safe_ticker for t in tickers])
        base = context.base_allocations
        weights[:, risky] = base[risky] * scale[:, None]
        safe_idx = context.ticker_index(self.safe_ticker) if self.safe_ticker else None
        if safe_idx is not None and not risky[safe_idx]:
            freed = base[risky].sum() - weights[:, risky].sum(axis=1)
            weights[:, safe_idx] = np.maximum(base[safe_idx] + freed, 0.0)
        return weights / weights.sum(axis=1, keepdims=True)


class DrawdownProtectionStrategy(AllocationStrategy):
    """Switch to `risk_off_allocation` when the base-weight portfolio's drawdown
    exceeds `dd_threshold`; switch back once it recovers to within
    `recovery_threshold`. The base-weight drawdown is used (not the actual
    portfolio's) because a portfolio sitting in bonds would never "recover"."""
    uses_rolling_stats = False

    def __init__(self, dd_threshold: float = 0.15,
                 risk_off_allocation: Dict[str, float] = None,
                 recovery_threshold: float = 0.05):
        super().__init__(name="Drawdown Protection")
        self.dd_threshold = dd_threshold
        self.risk_off_alloc = {k.upper(): v for k, v in (risk_off_allocation or {}).items()}
        self.recovery_threshold = recovery_threshold
        self._risk_off = None

    def reset(self, n_paths: int, asset_tickers: List[str]):
        self._risk_off = np.zeros(n_paths, dtype=bool)

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        weights = context.base()
        dd = (context.reference_drawdown if context.reference_drawdown is not None
              else context.portfolio_drawdown)
        if dd is None or not self.risk_off_alloc:
            return weights
        if self._risk_off is None or len(self._risk_off) != context.n_paths:
            self._risk_off = np.zeros(context.n_paths, dtype=bool)
        self._risk_off = np.where(self._risk_off, dd < -self.recovery_threshold, dd < -self.dd_threshold)
        if np.any(self._risk_off):
            off = np.array([self.risk_off_alloc.get(t.upper(), 0.0) for t in context.asset_tickers])
            if off.sum() > 0:
                weights[self._risk_off] = off / off.sum()
        return weights


class RelativeValueStrategy(AllocationStrategy):
    """When drawdown dispersion exceeds `threshold`, blend 50/50 toward the most
    beaten-down assets (mean reversion), capping any asset at `max_tilt`."""
    uses_rolling_stats = False

    def __init__(self, rebalance_threshold: float = 0.10, max_tilt: float = 0.50):
        super().__init__(name="Relative Value")
        self.threshold = rebalance_threshold
        self.max_tilt = max_tilt

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        weights = context.base()
        value = -context.current_drawdowns
        tilt = (value.max(axis=1) - value.min(axis=1)) > self.threshold
        if np.any(tilt):
            vs = value[tilt]
            vs = vs - vs.min(axis=1, keepdims=True) + 0.01
            blended = 0.5 * context.base_allocations + 0.5 * vs / vs.sum(axis=1, keepdims=True)
            blended = np.clip(blended, 0, self.max_tilt)
            weights[tilt] = blended / blended.sum(axis=1, keepdims=True)
        return weights


class DualMomentumStrategy(AllocationStrategy):
    """Absolute momentum / trend filter: hold base weights while the risky asset's
    trailing return beats the safe asset's (or 0 if the safe asset is not in the
    portfolio); otherwise hold 100% safe asset."""
    uses_rolling_stats = False

    def __init__(self, equity_ticker: str = "VOO", safe_ticker: str = "BND", lookback: int = 126):
        super().__init__(name=f"Dual Momentum ({equity_ticker}/{safe_ticker})")
        self.equity = equity_ticker.upper()
        self.safe = safe_ticker.upper()
        self.lookback = lookback
        self.lookback_days = lookback

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        weights = context.base()
        eq, safe = context.ticker_index(self.equity), context.ticker_index(self.safe)
        mom = context.trailing_return(self.lookback)
        if eq is None or safe is None or mom is None:
            return weights
        risk_off = mom[:, eq] <= mom[:, safe]
        weights[risk_off] = 0.0
        weights[risk_off, safe] = 1.0
        return weights


class CompositeStrategy(AllocationStrategy):
    """Weighted average of several strategies' targets."""

    def __init__(self, strategies: List[tuple]):
        super().__init__(name=f"Composite({', '.join(s.name for s, _ in strategies)})")
        self.strategies = strategies
        self.lookback_days = max((s.lookback_days for s, _ in strategies), default=0)
        self.uses_rolling_stats = any(s.uses_rolling_stats for s, _ in strategies)

    def reset(self, n_paths, asset_tickers):
        for s, _ in self.strategies:
            s.reset(n_paths, asset_tickers)

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        total = sum(w for _, w in self.strategies)
        combined = sum(s.get_allocation(context) * (w / total) for s, w in self.strategies)
        return combined / combined.sum(axis=1, keepdims=True)


class ConditionalStrategy(AllocationStrategy):
    """condition(context) -> bool (paths,) chooses between two strategies."""

    def __init__(self, condition: Callable, strategy_if_true: AllocationStrategy,
                 strategy_if_false: AllocationStrategy, name: str = "Conditional"):
        super().__init__(name=name)
        self.condition = condition
        self.true_strategy = strategy_if_true
        self.false_strategy = strategy_if_false
        self.lookback_days = max(strategy_if_true.lookback_days, strategy_if_false.lookback_days)

    def reset(self, n_paths, asset_tickers):
        self.true_strategy.reset(n_paths, asset_tickers)
        self.false_strategy.reset(n_paths, asset_tickers)

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        mask = np.asarray(self.condition(context), dtype=bool)
        return np.where(mask[:, None], self.true_strategy.get_allocation(context),
                        self.false_strategy.get_allocation(context))


# ============================================================================
# Registry: the only place config `type` strings map to classes.
# Param names match the documented StrategyConfig params.
# ============================================================================

def _p(params, *names, default=None):
    for n in names:
        if n in params:
            return params[n]
    return default


STRATEGY_BUILDERS: Dict[str, Callable[[Dict], AllocationStrategy]] = {
    'static': lambda p: StaticAllocationStrategy(),
    'buy_the_dip': lambda p: BuyTheDipStrategy(
        target_ticker=_p(p, 'target_ticker', default='VOO'),
        threshold=_p(p, 'threshold', default=0.10),
        aggressive_weight=_p(p, 'aggressive_weight', default=0.80)),
    'momentum': lambda p: MomentumStrategy(
        momentum_lookback=_p(p, 'lookback', 'momentum_lookback', default=63),
        tilt_strength=_p(p, 'tilt_strength', default=0.5),
        min_weight=_p(p, 'min_weight', default=0.05)),
    'volatility_target': lambda p: VolatilityTargetStrategy(
        target_vol=_p(p, 'target_vol', default=0.15),
        vol_lookback=_p(p, 'lookback', 'vol_lookback', default=21),
        equity_tickers=_p(p, 'equity_tickers', default=[]),
        safe_ticker=_p(p, 'safe_ticker', default=None)),
    'drawdown_protection': lambda p: DrawdownProtectionStrategy(
        dd_threshold=_p(p, 'threshold', 'dd_threshold', default=0.15),
        risk_off_allocation=_p(p, 'risk_off_allocation', default={}),
        recovery_threshold=_p(p, 'recovery_threshold', default=0.05)),
    'relative_value': lambda p: RelativeValueStrategy(
        rebalance_threshold=_p(p, 'threshold', 'rebalance_threshold', default=0.10),
        max_tilt=_p(p, 'max_tilt', default=0.50)),
    'crypto_opportunistic': lambda p: CryptoOpportunisticStrategy(
        crypto_ticker=_p(p, 'crypto_ticker', default='BTC-USD'),
        dip_threshold=_p(p, 'dip_threshold', default=0.25),
        normal_weight=_p(p, 'normal_weight', default=None),
        dip_weight=_p(p, 'dip_weight', default=0.40)),
    'dual_momentum': lambda p: DualMomentumStrategy(
        equity_ticker=_p(p, 'equity_ticker', default='VOO'),
        safe_ticker=_p(p, 'safe_ticker', default='BND'),
        lookback=_p(p, 'lookback', default=126)),
}


def create_strategy(strategy_type: str, params: Optional[Dict] = None,
                    name: Optional[str] = None) -> AllocationStrategy:
    if strategy_type not in STRATEGY_BUILDERS:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Valid: {sorted(STRATEGY_BUILDERS)}")
    strategy = STRATEGY_BUILDERS[strategy_type](params or {})
    if name:
        strategy.name = name
    return strategy
