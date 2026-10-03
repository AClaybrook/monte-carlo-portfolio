"""
The report shows the right numbers.

Runs main.py end to end on synthetic prices, then parses the HTML and checks:
1. table numbers against values computed independently from the raw prices
   with the naive reference implementation (tests/reference_impl.py);
2. every chart against the table that reports the same quantity.
"""
import base64
import json
import os
import re
import subprocess
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference_impl as ref
from synthetic_data import synthetic_prices

START, END = date(2016, 3, 1), date(2024, 11, 29)
RF = 0.02
CONFIG = f'''
from run_config import RunConfig, PortfolioConfig, SimulationConfig
config = RunConfig(
    name="Numbers check",
    benchmark_ticker='VOO',
    portfolios=[
        PortfolioConfig(name='VOO only', allocations={{'VOO': 1.0}}),
        PortfolioConfig(name='60/40 annual', allocations={{'VOO': 0.6, 'BND': 0.4}}, rebalance='annual'),
        PortfolioConfig(name='Crypto mix quarterly', allocations={{'QQQ': 0.7, 'BTC-USD': 0.3}},
                        rebalance='quarterly'),
    ],
    simulation=SimulationConfig(initial_capital=10000, years=4, simulations=400, seed=5,
                                start_date='{START}', end_date='{END}', risk_free_rate={RF},
                                contribution_amount=300, contribution_frequency='monthly',
                                rebalance='none'),
)
'''
PORTFOLIOS = {
    'VOO only': ({'VOO': 1.0}, 'none'),
    '60/40 annual': ({'VOO': 0.6, 'BND': 0.4}, 'annual'),
    'Crypto mix quarterly': ({'QQQ': 0.7, 'BTC-USD': 0.3}, 'quarterly'),
}


class Tables(HTMLParser):
    """Collect every <table> as {'headers': [...], 'rows': [[cell texts]]}."""

    def __init__(self):
        super().__init__()
        self.tables, self._row, self._cell, self._in_head = [], None, None, False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.tables.append({'headers': [], 'rows': []})
        elif tag == 'thead':
            self._in_head = True
        elif tag == 'tr':
            self._row = []
        elif tag in ('td', 'th'):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == 'thead':
            self._in_head = False
        elif tag in ('td', 'th') and self._cell is not None:
            self._row.append([t for t in self._cell if t.strip()])
            self._cell = None
        elif tag == 'tr' and self._row is not None:
            if self._in_head:
                self.tables[-1]['headers'] = [c[0] if c else '' for c in self._row]
            else:
                self.tables[-1]['rows'].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data.strip())


def num(cell):
    """'$1,234' -> 1234.0, '12.34%' -> 0.1234, '0.56' -> 0.56, '—' -> nan."""
    text = cell[0] if isinstance(cell, list) else cell
    if text in ('—', ''):
        return np.nan
    v = float(text.replace('$', '').replace(',', '').replace('%', ''))
    return v / 100 if text.endswith('%') else v


def table(tables, first_header_contains):
    for t in tables:
        if any(first_header_contains == h for h in t['headers']):
            return t
    raise KeyError(first_header_contains)


def rows_by_name(t):
    return {r[0][0]: dict(zip(t['headers'], r)) for r in t['rows']}


def decode(v):
    """Plotly 6 serializes arrays as {'dtype', 'bdata'}; plotly 5 as lists."""
    if isinstance(v, dict) and 'bdata' in v:
        arr = np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v['dtype']))
        return arr.reshape(v['shape']) if 'shape' in v and isinstance(v['shape'], list) else arr
    return np.asarray(v, dtype=object if v and isinstance(v[0], str) else float) if isinstance(v, list) else v


@pytest.fixture(scope='module')
def report(tmp_path_factory):
    cfg = tmp_path_factory.mktemp('cfg') / 'numbers_config.py'
    cfg.write_text(CONFIG)
    proc = subprocess.run([sys.executable, 'main.py', str(cfg), '--synthetic', '--no-optimize'], cwd=ROOT,
                          capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    path = Path(ROOT, re.search(r'Report saved to: (\S+)', proc.stdout).group(1))
    page = path.read_text()
    path.unlink()
    parser = Tables()
    parser.feed(page)
    figs = json.loads(re.search(r'const FIGS = (\{.*?\});\nconst CMAP', page, re.S).group(1))
    return parser.tables, figs


@pytest.fixture(scope='module')
def independent():
    """Every portfolio recomputed from raw synthetic prices with the reference code."""
    tickers = {'VOO', 'BND', 'QQQ', 'BTC-USD'}
    raw = {t: synthetic_prices(t, START, END) for t in tickers}
    start = max(s.index.min() for s in raw.values())
    end = min(s.index.max() for s in raw.values())
    out = {}
    bench = None
    for name, (alloc, freq) in PORTFOLIOS.items():
        prices = pd.concat([raw[t].rename(t) for t in alloc], axis=1, join='inner').dropna()
        prices = prices[(prices.index >= start) & (prices.index <= end)]
        dates = prices.index
        contrib_dates = ref.first_trading_days(dates, 'monthly')
        bal, twr, _ = ref.reference_backtest(prices, list(alloc.values()), 10000, 300.0, contrib_dates,
                                            ref.first_trading_days(dates, freq))
        flows = ([(dates[0], -10000.0)] + [(d, -300.0) for d in sorted(contrib_dates)]
                 + [(dates[-1], bal.iloc[-1])])
        monthly = ref.monthly_returns(twr)
        years = ref.calendar_year_returns(twr)
        out[name] = {'End': bal.iloc[-1], 'Contrib': 300.0 * len(contrib_dates), 'CAGR': ref.cagr(twr),
                     'IRR': ref.irr_bisect(flows), 'Stdev': ref.stdev(monthly) * np.sqrt(12),
                     'Sharpe': ref.sharpe(monthly, RF), 'Sortino': ref.sortino(monthly, RF),
                     'MaxDD': ref.max_drawdown(list(twr.values)), 'years': years, 'twr': twr}
    return out


# --------------------------------------------------------------------------
# 1. Tables vs independent computation
# --------------------------------------------------------------------------

@pytest.mark.parametrize('name', list(PORTFOLIOS))
def test_summary_table_matches_raw_prices(report, independent, name):
    tables, _ = report
    row = rows_by_name(table(tables, 'Final balance'))[name]
    exp = independent[name]
    assert num(row['Final balance']) == pytest.approx(exp['End'], abs=0.51)
    assert num(row['Contributions']) == pytest.approx(exp['Contrib'], abs=0.51)
    for col, key in (('CAGR', 'CAGR'), ('IRR', 'IRR'), ('Stdev', 'Stdev'), ('Max drawdown', 'MaxDD')):
        assert num(row[col]) == pytest.approx(exp[key], abs=0.00005 + 1e-9), col   # shown to 0.01%
    for col, key in (('Sharpe', 'Sharpe'), ('Sortino', 'Sortino')):
        assert num(row[col]) == pytest.approx(exp[key], abs=0.0051), col          # shown to 0.01


def test_annual_table_matches_raw_prices(report, independent):
    tables, _ = report
    t = table(tables, 'Year')
    for r in t['rows']:
        year = int(r[0][0].rstrip('†'))
        for name, cell in zip(t['headers'][1:], r[1:]):
            if name in independent:
                assert num(cell) == pytest.approx(independent[name]['years'][year], abs=0.00005), (name, year)


def test_largest_drawdown_equals_max_drawdown(report):
    tables, _ = report
    summary = rows_by_name(table(tables, 'Final balance'))
    dd = table(tables, 'Rank')
    current = None
    for r in dd['rows']:
        if r[0]:
            current = r[0][0]
        if num(r[1]) == 1:
            assert num(r[5]) == pytest.approx(num(summary[current]['Max drawdown']), abs=1e-9), current


# --------------------------------------------------------------------------
# 2. Charts vs tables
# --------------------------------------------------------------------------

def traces(figs, fig_id, name=None, meta=None):
    out = [t for t in figs[f'c-{fig_id}']['data']
           if (name is None or t.get('name') == name) and (meta is None or str(t.get('meta')) == meta)]
    assert out, (fig_id, name, meta)
    return out


def test_growth_chart_ends_at_final_balance(report):
    tables, figs = report
    for name, row in rows_by_name(table(tables, 'Final balance')).items():
        y = decode(traces(figs, 'growth', name)[0]['y'])
        assert y[-1] == pytest.approx(num(row['Final balance']), abs=0.51), name
        assert y[0] == pytest.approx(num(row['Initial']), abs=0.51), name


def test_drawdown_chart_matches_table(report):
    tables, figs = report
    for name, row in rows_by_name(table(tables, 'Final balance')).items():
        y = decode(traces(figs, 'drawdowns', name)[0]['y'])
        assert y.min() == pytest.approx(num(row['Max drawdown']), abs=0.00005), name
        assert y.max() <= 1e-12


def test_annual_bars_match_annual_table(report):
    tables, figs = report
    t = table(tables, 'Year')
    for col, name in enumerate(t['headers'][1:], start=1):
        bars = traces(figs, 'annual', name)[0]
        x, y = decode(bars['x']), decode(bars['y'])
        table_vals = {r[0][0]: num(r[col]) for r in t['rows'] if r[col] and r[col][0] != '—'}
        for label, v in zip(x, y):
            assert v == pytest.approx(table_vals[label], abs=0.00005), (name, label)


def test_monthly_heatmap_compounds_to_annual_returns(report):
    tables, figs = report
    t = table(tables, 'Year')
    labels = t['headers'][1:]
    for i, name in enumerate(labels):
        hm = traces(figs, 'monthly', meta=str(i))[0]
        z = np.array(decode(hm['z']), dtype=float).reshape(len(decode(hm['y'])), 12)
        for year, months in zip(decode(hm['y']), z):
            annual = np.prod(1 + months[~np.isnan(months)]) - 1
            row = next(r for r in t['rows'] if r[0][0].rstrip('†') == str(year))
            assert annual == pytest.approx(num(row[i + 1]), abs=0.00005), (name, year)


def test_risk_return_scatter_matches_table(report):
    tables, figs = report
    for name, row in rows_by_name(table(tables, 'Final balance')).items():
        pt = traces(figs, 'riskret', name)[0]
        assert decode(pt['x'])[0] == pytest.approx(num(row['Stdev']), abs=0.00005)
        assert decode(pt['y'])[0] == pytest.approx(num(row['CAGR']), abs=0.00005)


def test_monte_carlo_charts_match_table(report):
    tables, figs = report
    mc = rows_by_name(table(tables, 'Balance p50'))
    for i, (name, row) in enumerate(mc.items()):
        median_trace = [t for t in traces(figs, 'mcfan', meta=str(i)) if t.get('name') == 'Median'][0]
        assert decode(median_trace['y'])[-1] == pytest.approx(num(row['Balance p50']), abs=0.51), name
        assert decode(traces(figs, 'mcmedian', name)[0]['y'])[-1] == pytest.approx(num(row['Balance p50']), abs=0.51)
        box = traces(figs, 'mcbox', name)[0]
        assert decode(box['median'])[0] == pytest.approx(num(row['Median CAGR']), abs=0.00005)
        lo, hi = row['CAGR p10–p90'][0].split(' to ')
        assert decode(box['lowerfence'])[0] == pytest.approx(num(lo), abs=0.0005)
        assert decode(box['upperfence'])[0] == pytest.approx(num(hi), abs=0.0005)
        loss = decode(traces(figs, 'mcloss', name)[0]['y'])
        assert loss[-1] == pytest.approx(num(row['P(loss)']), abs=0.0005), name
        # Percentile columns are ordered
        bal = [num(row[f'Balance p{q}']) for q in (10, 25, 50, 75, 90)]
        assert bal == sorted(bal)


def test_contributions_line_in_growth_chart(report, independent):
    _, figs = report
    invested = decode(traces(figs, 'growth', 'Invested capital')[0]['y'])
    assert invested[0] == pytest.approx(10000)
    assert invested[-1] == pytest.approx(10000 + independent['VOO only']['Contrib'])
