"""
Deterministic synthetic market data for offline development and tests.

SyntheticDataManager mimics the parts of DataManager that main.py uses, so the
full pipeline can run with no network (`python main.py --synthetic`).

Prices are generated on a 7-day calendar from a regime-switching market factor
(bull/bear), then sampled on business days for exchange-traded tickers. Crypto
tickers ("-USD") keep all 7 days, which reproduces the real-world alignment
issue of mixing 24/7 and exchange calendars. Leveraged ETFs are built as daily
3x of their underlying minus fees, so volatility decay is realistic.

The numbers are plausible, NOT historical. Never use them for decisions.
"""
import zlib
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CALENDAR_START = date(2000, 1, 1)
CALENDAR_END = date(2026, 12, 31)

# ticker -> (annual drift, market beta, idiosyncratic annual vol)
_PROFILES = {
    'VOO': (0.10, 1.00, 0.02), 'SPY': (0.10, 1.00, 0.02), 'IVV': (0.10, 1.00, 0.02),
    'VFINX': (0.10, 1.00, 0.02), 'VTI': (0.10, 1.02, 0.03),
    'QQQ': (0.13, 1.20, 0.08), 'VGT': (0.14, 1.25, 0.09), 'SMH': (0.15, 1.40, 0.15),
    'VXUS': (0.06, 0.90, 0.08), 'VEA': (0.06, 0.90, 0.08), 'VWO': (0.06, 0.95, 0.12),
    'AVUV': (0.10, 1.10, 0.10), 'AVDV': (0.08, 1.00, 0.10),
    'BND': (0.035, -0.05, 0.05), 'AGG': (0.035, -0.05, 0.05),
    'IEF': (0.035, -0.12, 0.07), 'TLT': (0.04, -0.20, 0.13),
    'SHV': (0.02, 0.0, 0.004), 'BIL': (0.02, 0.0, 0.004), 'SGOV': (0.02, 0.0, 0.004),
    'GLD': (0.05, 0.05, 0.15), 'DBC': (0.03, 0.35, 0.16),
    'BTC-USD': (0.40, 1.30, 0.60), 'ETH-USD': (0.45, 1.50, 0.75), 'GBTC': (0.35, 1.30, 0.62),
}

# leveraged ticker -> (underlying, leverage, annual expense)
_LEVERAGED = {
    'SPXL': ('VOO', 3.0, 0.01), 'UPRO': ('VOO', 3.0, 0.01), 'SSO': ('VOO', 2.0, 0.009),
    'TQQQ': ('QQQ', 3.0, 0.01), 'QLD': ('QQQ', 2.0, 0.009),
    'SOXL': ('SMH', 3.0, 0.01), 'TMF': ('TLT', 3.0, 0.01),
}

_INCEPTION = {
    'BTC-USD': date(2014, 9, 17), 'ETH-USD': date(2017, 11, 9), 'GBTC': date(2015, 5, 5),
    'TQQQ': date(2010, 2, 11), 'SPXL': date(2008, 11, 5), 'UPRO': date(2009, 6, 25),
    'TMF': date(2009, 4, 16), 'SOXL': date(2010, 3, 11), 'VOO': date(2010, 9, 7),
    'BND': date(2007, 4, 3), 'VXUS': date(2011, 1, 26), 'AVUV': date(2019, 9, 24),
    'SGOV': date(2020, 5, 26),
}

_cache: Dict[str, pd.Series] = {}


def _calendar() -> pd.DatetimeIndex:
    return pd.date_range(CALENDAR_START, CALENDAR_END, freq='D')


def _market_factor() -> np.ndarray:
    """Daily simple returns of a two-regime market (bull 10%/15%, bear -30%/35%)."""
    if '__MKT__' in _cache:
        return _cache['__MKT__'].values
    rng = np.random.default_rng(12345)
    n = len(_calendar())
    dt = 1 / 365
    mu = {0: 0.13, 1: -0.30}
    vol = {0: 0.14, 1: 0.32}
    p_switch = {0: 1 / (365 * 4), 1: 1 / 180}   # ~4y bulls, ~6m bears
    regime = 0
    out = np.empty(n)
    for i in range(n):
        if rng.random() < p_switch[regime]:
            regime = 1 - regime
        out[i] = mu[regime] * dt + vol[regime] * np.sqrt(dt) * rng.standard_normal()
    _cache['__MKT__'] = pd.Series(out, index=_calendar())
    return out


def _daily_returns(ticker: str) -> pd.Series:
    """7-day-calendar simple returns for a ticker."""
    key = f'R:{ticker}'
    if key in _cache:
        return _cache[key]

    cal = _calendar()
    dt = 1 / 365
    if ticker in _LEVERAGED:
        base, lev, fee = _LEVERAGED[ticker]
        r = lev * _daily_returns(base).values - fee * dt
    else:
        mkt = _market_factor()
        mkt_mu = mkt.mean() / dt
        drift, beta, idio = _PROFILES.get(ticker, (0.08, 1.0, 0.15))
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        alpha = drift - beta * mkt_mu
        r = alpha * dt + beta * mkt + idio * np.sqrt(dt) * rng.standard_normal(len(cal))
    r = np.maximum(r, -0.95)
    s = pd.Series(r, index=cal)
    _cache[key] = s
    return s


def synthetic_prices(ticker: str, start: Optional[date] = None,
                     end: Optional[date] = None) -> pd.Series:
    """Adjusted close series on the ticker's trading calendar."""
    ticker = ticker.upper()
    r = _daily_returns(ticker)
    prices = 100.0 * (1 + r).cumprod()
    if not ticker.endswith('-USD'):
        prices = prices[prices.index.dayofweek < 5]
    first = max(start or CALENDAR_START, _INCEPTION.get(ticker, CALENDAR_START))
    last = min(end or CALENDAR_END, CALENDAR_END)
    prices = prices[(prices.index >= pd.Timestamp(first)) & (prices.index <= pd.Timestamp(last))]
    return prices.rename(ticker)


def synthetic_ohlcv(ticker: str, start: Optional[date] = None,
                    end: Optional[date] = None) -> pd.DataFrame:
    p = synthetic_prices(ticker, start, end)
    df = pd.DataFrame({'Open': p, 'High': p, 'Low': p, 'Close': p, 'Adj Close': p,
                       'Volume': 1_000_000})
    df.index.name = 'Date'
    return df


class SyntheticDataManager:
    """Drop-in stand-in for DataManager backed by synthetic_ohlcv()."""

    def __init__(self, *args, **kwargs):
        self.data_source = 'synthetic'

    def get_data(self, ticker: str, start_date=None, end_date=None, force_update=False):
        return synthetic_ohlcv(ticker, _to_date(start_date), _to_date(end_date))

    def bulk_download(self, tickers: List[str], start_date=None, end_date=None,
                      force_update: bool = False, sequential: bool = False):
        out = {}
        for t in tickers:
            df = self.get_data(t, start_date, end_date)
            if not df.empty:
                out[t.upper()] = df
        return out

    def _get_from_db(self, ticker, start_date, end_date):
        df = self.get_data(ticker, start_date, end_date)
        return None if df.empty else df

    def get_latest_cached_date(self, tickers=None):
        return CALENDAR_END

    def get_data_coverage_report(self, tickers):
        rows = []
        for t in tickers:
            p = synthetic_prices(t)
            rows.append({'ticker': t, 'start': p.index.min().date(),
                         'end': p.index.max().date(), 'rows': len(p)})
        return pd.DataFrame(rows)

    def close(self):
        pass


def _to_date(d) -> Optional[date]:
    if d is None:
        return None
    return pd.Timestamp(d).date()
