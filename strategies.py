"""
strategies.py - Enhanced Dynamic Allocation Strategies

Provides a rich framework for implementing and testing dynamic allocation strategies
that respond to market conditions (drawdowns, momentum, volatility, etc.)
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    NEUTRAL = "neutral"


@dataclass
class MarketContext:
    """
    Rich market context passed to strategies for decision making.

    This provides strategies with everything they need to make informed decisions:
    - Current portfolio state
    - Historical price/return data
    - Technical indicators
    - Cross-asset relationships
    """
    # Current State (shape: Batch x Assets)
    current_holdings: np.ndarray          # Dollar value per asset
    current_drawdowns: np.ndarray         # Drawdown from peak per asset

    # Base Configuration
    base_allocations: np.ndarray          # Target weights
    asset_tickers: List[str]              # Ticker names for reference

    # Time Context
    current_day: int                      # Day in simulation
    total_days: int                       # Total simulation length

    # Rolling Statistics (shape: Batch x Assets) - computed over lookback window
    rolling_returns: Optional[np.ndarray] = None      # Annualized returns
    rolling_volatility: Optional[np.ndarray] = None   # Annualized vol
    rolling_sharpe: Optional[np.ndarray] = None       # Sharpe ratio
    momentum_score: Optional[np.ndarray] = None       # Momentum indicator

    # Cross-Asset (shape: Batch)
    portfolio_drawdown: Optional[np.ndarray] = None   # Total portfolio DD
    market_regime: Optional[np.ndarray] = None        # Regime classification

    # Price History (for complex strategies) - shape: (lookback_days, assets)
    # Only populated if strategy requests it
    price_history: Optional[np.ndarray] = None


class AllocationStrategy(ABC):
    """
    Base class for all allocation strategies.

    Strategies determine how new contributions (DCA) are allocated,
    potentially overriding the base allocation based on market conditions.
    """

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.requires_history = False  # Set True if strategy needs price_history
        self.lookback_days = 0         # How many days of history needed

    @abstractmethod
    def get_allocation(self, context: MarketContext) -> np.ndarray:
        """
        Determine allocation weights for new contribution.

        Parameters:
        -----------
        context : MarketContext
            Rich market context with current state and indicators

        Returns:
        --------
        np.ndarray : Shape (Batch, Assets) - allocation weights summing to 1.0
        """
        pass

    def get_config_summary(self) -> Dict:
        """Return strategy configuration for logging"""
        return {"name": self.name}


class StaticAllocationStrategy(AllocationStrategy):
    """Standard DCA: Always allocate according to fixed portfolio weights."""

    def __init__(self):
        super().__init__(name="Static DCA")

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size = context.current_holdings.shape[0]
        return np.tile(context.base_allocations, (batch_size, 1))


class BuyTheDipStrategy(AllocationStrategy):
    """
    Allocates more heavily into assets experiencing significant drawdowns.

    Parameters:
    -----------
    target_ticker : str
        Ticker symbol to monitor for dips
    threshold : float
        Drawdown threshold to trigger (e.g., 0.10 = 10% drop)
    aggressive_weight : float
        Weight to allocate to dipped asset (e.g., 0.8 = 80%)
    """

    def __init__(self, target_ticker: str, threshold: float = 0.10,
                 aggressive_weight: float = 0.80):
        super().__init__(name=f"Buy the Dip ({target_ticker})")
        self.target_ticker = target_ticker.upper()
        self.threshold = threshold
        self.aggressive_weight = aggressive_weight
        self._target_idx = None  # Set dynamically

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size, n_assets = context.current_holdings.shape

        # Find target index if not set
        if self._target_idx is None:
            try:
                self._target_idx = [t.upper() for t in context.asset_tickers].index(self.target_ticker)
            except ValueError:
                print(f"Warning: {self.target_ticker} not in portfolio, using static allocation")
                return np.tile(context.base_allocations, (batch_size, 1))

        # Default to base allocation
        weights = np.tile(context.base_allocations, (batch_size, 1))

        # Check for dip condition
        target_dd = context.current_drawdowns[:, self._target_idx]
        dip_mask = target_dd < -self.threshold

        if np.any(dip_mask):
            # Reallocate for dipped scenarios
            weights[dip_mask] = 0
            weights[dip_mask, self._target_idx] = self.aggressive_weight

            # Distribute remaining weight proportionally
            remaining = 1.0 - self.aggressive_weight
            base_others = np.sum(context.base_allocations) - context.base_allocations[self._target_idx]

            if base_others > 0:
                for i in range(n_assets):
                    if i != self._target_idx:
                        w = (context.base_allocations[i] / base_others) * remaining
                        weights[dip_mask, i] = w

        return weights

    def get_config_summary(self) -> Dict:
        return {
            "name": self.name,
            "target": self.target_ticker,
            "threshold": f"{self.threshold*100:.0f}%",
            "aggressive_weight": f"{self.aggressive_weight*100:.0f}%"
        }


class MomentumStrategy(AllocationStrategy):
    """
    Tilts allocation toward assets with positive momentum.

    Overweights assets with strong recent returns, underweights laggards.

    Parameters:
    -----------
    momentum_lookback : int
        Days to calculate momentum over
    tilt_strength : float
        How aggressively to tilt (0 = none, 1 = full momentum weighting)
    min_weight : float
        Minimum weight for any asset (prevents going to 0)
    """

    def __init__(self, momentum_lookback: int = 63, tilt_strength: float = 0.5,
                 min_weight: float = 0.05):
        super().__init__(name="Momentum Tilt")
        self.lookback = momentum_lookback
        self.tilt_strength = tilt_strength
        self.min_weight = min_weight
        self.requires_history = True
        self.lookback_days = momentum_lookback

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size, n_assets = context.current_holdings.shape

        if context.momentum_score is None:
            # Fallback to static if no momentum data
            return np.tile(context.base_allocations, (batch_size, 1))

        # Normalize momentum scores to weights
        # Shift to positive, then normalize
        mom = context.momentum_score  # (Batch, Assets)
        mom_shifted = mom - mom.min(axis=1, keepdims=True) + 1e-6
        mom_weights = mom_shifted / mom_shifted.sum(axis=1, keepdims=True)

        # Blend with base allocation
        base = np.tile(context.base_allocations, (batch_size, 1))
        blended = (1 - self.tilt_strength) * base + self.tilt_strength * mom_weights

        # Enforce minimum weights
        blended = np.maximum(blended, self.min_weight)
        blended = blended / blended.sum(axis=1, keepdims=True)

        return blended


class VolatilityTargetStrategy(AllocationStrategy):
    """
    Adjusts allocation to maintain a target portfolio volatility.

    When realized vol is high, reduces equity exposure.
    When realized vol is low, increases exposure.

    Parameters:
    -----------
    target_vol : float
        Target annualized volatility (e.g., 0.15 = 15%)
    vol_lookback : int
        Days to calculate realized volatility
    equity_tickers : list
        Tickers considered "risky" assets
    safe_ticker : str
        Ticker to shift into when de-risking (e.g., 'BND', 'SHY')
    """

    def __init__(self, target_vol: float = 0.15, vol_lookback: int = 21,
                 equity_tickers: List[str] = None, safe_ticker: str = None):
        super().__init__(name="Volatility Target")
        self.target_vol = target_vol
        self.lookback = vol_lookback
        self.equity_tickers = [t.upper() for t in (equity_tickers or [])]
        self.safe_ticker = safe_ticker.upper() if safe_ticker else None
        self.requires_history = True
        self.lookback_days = vol_lookback

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size, n_assets = context.current_holdings.shape

        if context.rolling_volatility is None:
            return np.tile(context.base_allocations, (batch_size, 1))

        # Calculate portfolio vol (simplified: weighted sum of asset vols)
        base = context.base_allocations
        port_vol = np.sum(context.rolling_volatility * base, axis=1)  # (Batch,)

        # Scale factor to hit target
        scale = np.clip(self.target_vol / (port_vol + 1e-6), 0.5, 1.5)  # (Batch,)

        # For now, simple scaling of base allocation
        # More sophisticated: shift between equity and safe assets
        weights = np.tile(base, (batch_size, 1))

        # Scale risky assets down/up
        for i, ticker in enumerate(context.asset_tickers):
            if ticker.upper() in self.equity_tickers:
                weights[:, i] = weights[:, i] * scale

        # Renormalize
        weights = weights / weights.sum(axis=1, keepdims=True)

        return weights


class DrawdownProtectionStrategy(AllocationStrategy):
    """
    Reduces equity exposure when portfolio drawdown exceeds threshold.

    Implements a simple risk-off mechanism during market stress.

    Parameters:
    -----------
    dd_threshold : float
        Portfolio drawdown to trigger de-risking (e.g., 0.15 = 15%)
    risk_off_allocation : dict
        Target allocation during risk-off (e.g., {'BND': 0.6, 'VOO': 0.4})
    recovery_threshold : float
        Drawdown level to return to normal allocation
    """

    def __init__(self, dd_threshold: float = 0.15,
                 risk_off_allocation: Dict[str, float] = None,
                 recovery_threshold: float = 0.05):
        super().__init__(name="Drawdown Protection")
        self.dd_threshold = dd_threshold
        self.risk_off_alloc = risk_off_allocation or {}
        self.recovery_threshold = recovery_threshold

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size, n_assets = context.current_holdings.shape

        # Start with base allocation
        weights = np.tile(context.base_allocations, (batch_size, 1))

        if context.portfolio_drawdown is None:
            return weights

        # Check which simulations are in risk-off mode
        risk_off_mask = context.portfolio_drawdown < -self.dd_threshold

        if np.any(risk_off_mask) and self.risk_off_alloc:
            # Apply risk-off allocation
            for i, ticker in enumerate(context.asset_tickers):
                t_upper = ticker.upper()
                if t_upper in self.risk_off_alloc:
                    weights[risk_off_mask, i] = self.risk_off_alloc[t_upper]
                else:
                    weights[risk_off_mask, i] = 0

            # Normalize
            row_sums = weights[risk_off_mask].sum(axis=1, keepdims=True)
            weights[risk_off_mask] = weights[risk_off_mask] / np.where(row_sums > 0, row_sums, 1)

        return weights


class RelativeValueStrategy(AllocationStrategy):
    """
    Allocates more to assets that are "cheap" relative to their history.

    Uses drawdown as a proxy for relative value - bigger drawdowns = cheaper.
    Good for mean-reversion beliefs.

    Parameters:
    -----------
    rebalance_threshold : float
        Minimum drawdown difference to trigger rebalancing
    max_tilt : float
        Maximum overweight for any single asset
    """

    def __init__(self, rebalance_threshold: float = 0.10, max_tilt: float = 0.50):
        super().__init__(name="Relative Value")
        self.threshold = rebalance_threshold
        self.max_tilt = max_tilt

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size, n_assets = context.current_holdings.shape

        # Deeper drawdown = more attractive (mean reversion)
        # Convert drawdowns to "value scores" (more negative = higher score)
        value_scores = -context.current_drawdowns  # (Batch, Assets)

        # Only tilt if there's meaningful dispersion
        dd_range = value_scores.max(axis=1) - value_scores.min(axis=1)

        weights = np.tile(context.base_allocations, (batch_size, 1))

        # For sims with enough dispersion, tilt toward beaten-down assets
        tilt_mask = dd_range > self.threshold

        if np.any(tilt_mask):
            # Normalize value scores to weights
            vs = value_scores[tilt_mask]
            vs_shifted = vs - vs.min(axis=1, keepdims=True) + 0.01
            value_weights = vs_shifted / vs_shifted.sum(axis=1, keepdims=True)

            # Blend: 50% base, 50% value tilt
            base = context.base_allocations
            blended = 0.5 * base + 0.5 * value_weights

            # Cap tilts
            blended = np.clip(blended, 0, self.max_tilt)
            blended = blended / blended.sum(axis=1, keepdims=True)

            weights[tilt_mask] = blended

        return weights


class CompositeStrategy(AllocationStrategy):
    """
    Combines multiple strategies with configurable weights.

    Parameters:
    -----------
    strategies : list of (strategy, weight) tuples
        Strategies to combine and their relative weights
    """

    def __init__(self, strategies: List[tuple]):
        names = [s[0].name for s in strategies]
        super().__init__(name=f"Composite({', '.join(names)})")
        self.strategies = strategies

        # Set requirements based on children
        self.requires_history = any(s.requires_history for s, _ in strategies)
        self.lookback_days = max((s.lookback_days for s, _ in strategies), default=0)

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size, n_assets = context.current_holdings.shape

        total_weight = sum(w for _, w in self.strategies)
        combined = np.zeros((batch_size, n_assets))

        for strategy, weight in self.strategies:
            alloc = strategy.get_allocation(context)
            combined += alloc * (weight / total_weight)

        # Normalize
        combined = combined / combined.sum(axis=1, keepdims=True)

        return combined


class ConditionalStrategy(AllocationStrategy):
    """
    Switches between strategies based on market conditions.

    Parameters:
    -----------
    condition : callable
        Function(context) -> bool array indicating which condition is met
    strategy_if_true : AllocationStrategy
    strategy_if_false : AllocationStrategy
    """

    def __init__(self, condition: Callable,
                 strategy_if_true: AllocationStrategy,
                 strategy_if_false: AllocationStrategy,
                 name: str = "Conditional"):
        super().__init__(name=name)
        self.condition = condition
        self.true_strategy = strategy_if_true
        self.false_strategy = strategy_if_false

        self.requires_history = (strategy_if_true.requires_history or
                                  strategy_if_false.requires_history)
        self.lookback_days = max(strategy_if_true.lookback_days,
                                  strategy_if_false.lookback_days)

    def get_allocation(self, context: MarketContext) -> np.ndarray:
        batch_size, n_assets = context.current_holdings.shape

        # Evaluate condition
        mask = self.condition(context)  # (Batch,) bool

        # Get allocations from both strategies
        true_alloc = self.true_strategy.get_allocation(context)
        false_alloc = self.false_strategy.get_allocation(context)

        # Combine based on condition
        weights = np.where(mask[:, np.newaxis], true_alloc, false_alloc)

        return weights


# ============================================================================
# Factory Functions for Easy Strategy Creation
# ============================================================================

def create_btc_dip_buyer(btc_ticker: str = "BTC-USD",
                         threshold: float = 0.20,
                         aggressive_weight: float = 0.50) -> BuyTheDipStrategy:
    """
    Creates a strategy that buys Bitcoin aggressively during major dips.

    Default: When BTC drops 20%, allocate 50% of new money to it.
    """
    return BuyTheDipStrategy(
        target_ticker=btc_ticker,
        threshold=threshold,
        aggressive_weight=aggressive_weight
    )


def create_dual_momentum_strategy(equity_ticker: str = "VOO",
                                   safe_ticker: str = "BND",
                                   lookback: int = 126) -> ConditionalStrategy:
    """
    Classic dual momentum: Risk-on when equities have positive momentum,
    risk-off otherwise.
    """
    def positive_momentum(ctx: MarketContext) -> np.ndarray:
        if ctx.momentum_score is None:
            return np.ones(ctx.current_holdings.shape[0], dtype=bool)

        try:
            eq_idx = [t.upper() for t in ctx.asset_tickers].index(equity_ticker.upper())
            return ctx.momentum_score[:, eq_idx] > 0
        except ValueError:
            return np.ones(ctx.current_holdings.shape[0], dtype=bool)

    risk_on = StaticAllocationStrategy()
    risk_off = DrawdownProtectionStrategy(
        dd_threshold=0.0,  # Always "triggered"
        risk_off_allocation={safe_ticker.upper(): 1.0}
    )

    return ConditionalStrategy(
        condition=positive_momentum,
        strategy_if_true=risk_on,
        strategy_if_false=risk_off,
        name=f"Dual Momentum ({equity_ticker}/{safe_ticker})"
    )


def create_crypto_opportunistic_strategy(
    crypto_ticker: str = "BTC-USD",
    equity_ticker: str = "VOO",
    crypto_dip_threshold: float = 0.25,
    normal_crypto_weight: float = 0.10,
    dip_crypto_weight: float = 0.40
) -> AllocationStrategy:
    """
    Strategy that normally maintains small crypto allocation but
    increases significantly during major crypto drawdowns.

    Example: Normally 10% BTC, but 40% when BTC is down 25%+
    """

    class CryptoOpportunistic(AllocationStrategy):
        def __init__(self):
            super().__init__(name=f"Crypto Opportunistic ({crypto_ticker})")
            self.crypto = crypto_ticker.upper()
            self.equity = equity_ticker.upper()
            self.threshold = crypto_dip_threshold
            self.normal_weight = normal_crypto_weight
            self.dip_weight = dip_crypto_weight

        def get_allocation(self, context: MarketContext) -> np.ndarray:
            batch_size, n_assets = context.current_holdings.shape

            try:
                crypto_idx = [t.upper() for t in context.asset_tickers].index(self.crypto)
                equity_idx = [t.upper() for t in context.asset_tickers].index(self.equity)
            except ValueError:
                return np.tile(context.base_allocations, (batch_size, 1))

            weights = np.tile(context.base_allocations, (batch_size, 1))

            # Check crypto drawdown
            crypto_dd = context.current_drawdowns[:, crypto_idx]
            dip_mask = crypto_dd < -self.threshold

            if np.any(dip_mask):
                # During dip: increase crypto, decrease equity proportionally
                weights[dip_mask, crypto_idx] = self.dip_weight

                # Reduce other weights proportionally to fit
                other_sum = 1.0 - self.dip_weight
                original_other = 1.0 - context.base_allocations[crypto_idx]

                for i in range(n_assets):
                    if i != crypto_idx:
                        scale = other_sum / original_other if original_other > 0 else 0
                        weights[dip_mask, i] = context.base_allocations[i] * scale

            # Normal times: use normal crypto weight
            normal_mask = ~dip_mask
            if np.any(normal_mask):
                weights[normal_mask, crypto_idx] = self.normal_weight
                other_sum = 1.0 - self.normal_weight
                original_other = 1.0 - context.base_allocations[crypto_idx]

                for i in range(n_assets):
                    if i != crypto_idx:
                        scale = other_sum / original_other if original_other > 0 else 0
                        weights[normal_mask, i] = context.base_allocations[i] * scale

            return weights

        def get_config_summary(self) -> Dict:
            return {
                "name": self.name,
                "crypto_ticker": self.crypto,
                "dip_threshold": f"{self.threshold*100:.0f}%",
                "normal_weight": f"{self.normal_weight*100:.0f}%",
                "dip_weight": f"{self.dip_weight*100:.0f}%"
            }

    return CryptoOpportunistic()


# ============================================================================
# Strategy Registry for Config-Based Loading
# ============================================================================

STRATEGY_REGISTRY = {
    "static": StaticAllocationStrategy,
    "buy_the_dip": BuyTheDipStrategy,
    "momentum": MomentumStrategy,
    "volatility_target": VolatilityTargetStrategy,
    "drawdown_protection": DrawdownProtectionStrategy,
    "relative_value": RelativeValueStrategy,
}


def create_strategy_from_config(config: Dict) -> AllocationStrategy:
    """
    Factory function to create strategies from configuration dictionaries.

    Example config:
    {
        "type": "buy_the_dip",
        "params": {
            "target_ticker": "BTC-USD",
            "threshold": 0.20,
            "aggressive_weight": 0.50
        }
    }
    """
    strategy_type = config.get("type", "static")
    params = config.get("params", {})

    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return STRATEGY_REGISTRY[strategy_type](**params)