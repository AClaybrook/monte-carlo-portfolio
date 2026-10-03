"""
HTML report in the spirit of Portfolio Visualizer's backtest + Monte Carlo pages.

Rule: this module only presents. Every number comes from the backtester
(quant_analytics.compute_performance) or the simulator; nothing financial is
recomputed here, so the tables and charts cannot disagree with the engine.

Charts are Plotly figures serialized to JSON and rendered by a small script that
swaps light/dark colors and drives the per-portfolio selectors.
"""
import base64
import html
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs, get_plotlyjs_version

import quant_analytics as qa
from pv_compat import export_portfolio_csv, generate_mc_url, generate_pv_url

# Validated categorical palette (light, dark): fixed order, never cycled.
SERIES = [('#2a78d6', '#3987e5'), ('#eb6834', '#d95926'), ('#1baf7a', '#199e70'),
          ('#eda100', '#c98500'), ('#e87ba4', '#d55181'), ('#008300', '#008300'),
          ('#4a3aa7', '#9085e9'), ('#e34948', '#e66767')]
BENCHMARK = ('#52514e', '#c3c2b7')
OVERFLOW = ('#898781', '#898781')
OVERFLOW_DASHES = ['dot', 'dash', 'dashdot', 'longdash']

CHROME = {  # role: (light, dark)
    'surface': ('#fcfcfb', '#1a1a19'),
    'page': ('#f9f9f7', '#0d0d0d'),
    'text': ('#0b0b0b', '#ffffff'),
    'text2': ('#52514e', '#c3c2b7'),
    'muted': ('#898781', '#8a8a84'),
    'grid': ('#e1e0d9', '#2c2c2a'),
    'axis': ('#c3c2b7', '#383835'),
    'mid': ('#f0efec', '#383835'),
    'neg': ('#e34948', '#e66767'),
    'pos': ('#2a78d6', '#3987e5'),
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
PCTS = (10, 25, 50, 75, 90)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def pct(x, d=2):
    if _nan(x):
        return '—'
    v = round(x * 100, d)
    return f"{v + 0.0:,.{d}f}%"  # + 0.0 turns -0.0 into 0.0


def money(x):
    return '—' if _nan(x) else f"${x:,.0f}"


def num(x, d=2):
    return '—' if _nan(x) else f"{x:,.{d}f}"


def esc(s) -> str:
    return html.escape(str(s))


class PortfolioVisualizer:
    def __init__(self, simulator=None):
        self.simulator = simulator
        self._cmap: Dict[str, str] = {}

    # ------------------------------------------------------------------ colors

    def _c(self, pair) -> str:
        """Register a (light, dark) pair; return the light value for the figure."""
        light, dark = pair
        self._cmap[light] = dark
        return light

    def _wash(self, pair, alpha) -> str:
        def rgba(h):
            h = h.lstrip('#')
            return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"
        return self._c((rgba(pair[0]), rgba(pair[1])))

    def _chrome(self, role) -> str:
        return self._c(CHROME[role])

    def _assign_styles(self, items):
        styles, slot = [], 0
        for item in items:
            if item.get('is_benchmark'):
                styles.append({'pair': BENCHMARK, 'dash': 'solid', 'width': 1.5})
            elif slot < len(SERIES):
                styles.append({'pair': SERIES[slot], 'dash': 'solid', 'width': 2})
                slot += 1
            else:
                styles.append({'pair': OVERFLOW, 'width': 2,
                               'dash': OVERFLOW_DASHES[(slot - len(SERIES)) % len(OVERFLOW_DASHES)]})
                slot += 1
        return styles

    def _line(self, style):
        return dict(color=self._c(style['pair']), width=style['width'], dash=style['dash'])

    # ------------------------------------------------------------------ layout

    def _layout(self, height=380, yfmt=None, ytitle=None, ylog=False, xtitle=None,
                hover='x unified', legend=True, **extra):
        axis = dict(gridcolor=self._chrome('grid'), linecolor=self._chrome('axis'),
                    zerolinecolor=self._chrome('axis'), showline=True, zeroline=False,
                    tickfont=dict(color=self._chrome('muted')), automargin=True,
                    title=dict(font=dict(color=self._chrome('text2'))))
        layout = dict(
            height=height,
            margin=dict(l=8, r=16, t=8, b=8),
            paper_bgcolor=self._chrome('surface'),
            plot_bgcolor=self._chrome('surface'),
            font=dict(family=FONT, size=12, color=self._chrome('text2')),
            hovermode=hover,
            hoverlabel=dict(bgcolor=self._chrome('surface'), bordercolor=self._chrome('axis'),
                            font=dict(color=self._chrome('text'), family=FONT)),
            showlegend=legend,
            legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0, xanchor='left',
                        font=dict(color=self._chrome('text2')), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(axis, title=dict(axis['title'], text=xtitle)),
            yaxis=dict(axis, title=dict(axis['title'], text=ytitle), tickformat=yfmt),
        )
        if ylog:
            layout['yaxis'].update(type='log', dtick='D2')
            if yfmt == '$,.0f':
                layout['yaxis']['tickformat'] = '$~s'
        layout.update(extra)
        return layout

    @staticmethod
    def _fig_json(fig: go.Figure) -> dict:
        return json.loads(fig.to_json())

    # ------------------------------------------------------------------ charts

    def _growth(self, items, styles):
        fig = go.Figure()
        for i, (it, st) in enumerate(zip(items, styles)):
            bal = it['backtest']['balance']
            fig.add_trace(go.Scatter(x=bal.index, y=bal.values, name=it['label'], line=self._line(st),
                                     hovertemplate='%{y:$,.0f}'))
        ref = next((it['backtest'] for it in items if not it.get('is_benchmark')), items[0]['backtest'])
        if ref['metrics']['Total Contributions'] > 0:
            invested = ref['metrics']['Start Balance'] + ref['contributions'].reindex(
                ref['balance'].index).fillna(0).cumsum()
            fig.add_trace(go.Scatter(x=invested.index, y=invested.values, name='Invested capital',
                                     line=dict(color=self._chrome('muted'), width=1.5, shape='hv'),
                                     hovertemplate='%{y:$,.0f}'))
        fig.update_layout(**self._layout(height=440, yfmt='$,.0f', ytitle='Balance (log scale)', ylog=True))
        return fig

    def _annual_bars(self, items, styles):
        fig = go.Figure()
        for it, st in zip(items, styles):
            a = it['backtest']['annual_returns']
            labels = [f"{y}{'†' if p else ''}" for y, p in zip(a.index, a['partial'])]
            fig.add_trace(go.Bar(x=labels, y=a['return'].values, name=it['label'],
                                 marker=dict(color=self._c(st['pair']), line=dict(width=0)),
                                 hovertemplate='%{y:.2%}'))
        fig.update_layout(**self._layout(yfmt='.0%', ytitle='Calendar-year return',
                                         barmode='group', bargap=0.25, bargroupgap=0.08,
                                         barcornerradius=4))
        return fig

    def _drawdowns(self, items, styles):
        fig = go.Figure()
        for it, st in zip(items, styles):
            dd = it['backtest']['drawdowns']
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=it['label'],
                                     line=dict(self._line(st), width=1.5), hovertemplate='%{y:.2%}'))
        fig.update_layout(**self._layout(yfmt='.0%', ytitle='Drawdown from peak'))
        return fig

    def _rolling(self, items, styles):
        fig = go.Figure()
        for window in ('1y', '3y', '5y'):
            for it, st in zip(items, styles):
                r = it['backtest'][f'rolling_{window}'].dropna()
                fig.add_trace(go.Scatter(x=r.index, y=r.values, name=it['label'], meta=window,
                                         legendgroup=it['label'], showlegend=window == '1y',
                                         visible=window == '1y', line=self._line(st),
                                         hovertemplate='%{y:.2%}'))
        fig.update_layout(**self._layout(yfmt='.0%', ytitle='Annualized trailing return'))
        return fig

    def _risk_return(self, items, styles):
        fig = go.Figure()
        for it, st in zip(items, styles):
            m = it['backtest']['metrics']
            fig.add_trace(go.Scatter(
                x=[m['Stdev']], y=[m['CAGR']], name=it['label'], mode='markers+text',
                text=[it['label']], textposition='top center',
                textfont=dict(color=self._chrome('text2'), size=11),
                marker=dict(size=11, color=self._c(st['pair']),
                            line=dict(width=2, color=self._chrome('surface'))),
                hovertemplate='Stdev %{x:.2%}<br>CAGR %{y:.2%}<extra>%{text}</extra>'))
        fig.update_layout(**self._layout(height=420, yfmt='.0%', ytitle='CAGR', xtitle='Annualized stdev',
                                         hover='closest', legend=False,
                                         xaxis=dict(self._layout()['xaxis'], tickformat='.0%')))
        return fig

    def _diverging_scale(self):
        return [[0, self._chrome('neg')], [0.5, self._chrome('mid')], [1, self._chrome('pos')]]

    def _monthly_heatmap(self, items):
        fig = go.Figure()
        for i, it in enumerate(items):
            t = it['backtest']['monthly_table']
            z = t.drop(columns='Year')
            text = [[pct(v, 1) if not _nan(v) else '' for v in row] for row in z.values]
            fig.add_trace(go.Heatmap(
                z=z.values, x=list(z.columns), y=[str(y) for y in z.index], meta=str(i),
                visible=i == 0, zmid=0, zmin=-0.15, zmax=0.15, colorscale=self._diverging_scale(),
                text=text, texttemplate='%{text}', textfont=dict(size=10), xgap=2, ygap=2,
                colorbar=dict(tickformat='.0%', outlinewidth=0, thickness=10,
                              tickfont=dict(color=self._chrome('muted'))),
                hovertemplate='%{y} %{x}: %{z:.2%}<extra></extra>'))
        n_years = max(len(it['backtest']['monthly_table']) for it in items)
        fig.update_layout(**self._layout(height=max(260, 26 * n_years + 60), hover='closest', legend=False,
                                         yaxis=dict(self._layout()['yaxis'], autorange='reversed',
                                                    type='category', showgrid=False),
                                         xaxis=dict(self._layout()['xaxis'], showgrid=False, side='top')))
        return fig

    def _correlation(self, items):
        prices = {}
        for it in items:
            for col, s in it['backtest']['asset_prices'].items():
                if col not in prices or len(s) > len(prices[col]):
                    prices[col] = s
        if len(prices) < 2:
            return None
        rets = pd.DataFrame({k: qa.monthly_returns(v) for k, v in prices.items()})
        corr = rets.corr(min_periods=12)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, zmid=0,
            colorscale=self._diverging_scale(), xgap=2, ygap=2,
            text=[[num(v) for v in row] for row in corr.values], texttemplate='%{text}',
            colorbar=dict(outlinewidth=0, thickness=10, tickfont=dict(color=self._chrome('muted'))),
            hovertemplate='%{y} / %{x}: %{z:.2f}<extra></extra>'))
        size = max(300, 48 * len(corr) + 80)
        fig.update_layout(**self._layout(height=size, hover='closest', legend=False,
                                         yaxis=dict(self._layout()['yaxis'], autorange='reversed', showgrid=False),
                                         xaxis=dict(self._layout()['xaxis'], showgrid=False, side='top')))
        return fig, corr

    def _allocation(self, items):
        tickers = []
        for it in items:
            for t in it['backtest']['weights'].columns:
                if t not in tickers:
                    tickers.append(t)
        colors = {t: (SERIES[i] if i < len(SERIES) else OVERFLOW) for i, t in enumerate(tickers)}
        fig = go.Figure()
        shapes = []
        for i, it in enumerate(items):
            w = it['backtest']['weights'].resample('W').last().dropna(how='all')
            for t in w.columns:
                fig.add_trace(go.Scatter(
                    x=w.index, y=w[t].values, name=t, meta=str(i), visible=i == 0, stackgroup=f's{i}',
                    legendgroup=t, line=dict(width=1, color=self._chrome('surface')),
                    fillcolor=self._c(colors[t]), hovertemplate='%{y:.1%}'))
            for e in it['backtest']['events']:
                if e['type'] == 'rebalance' and e['trigger'] in ('signal', 'threshold'):
                    shapes.append(dict(type='line', xref='x', yref='paper', x0=e['date'], x1=e['date'],
                                       y0=0, y1=1, name=str(i), visible=i == 0,
                                       line=dict(color=self._chrome('text2'), width=1)))
        fig.update_layout(**self._layout(yfmt='.0%', ytitle='Weight', shapes=shapes,
                                         yaxis=dict(self._layout()['yaxis'], range=[0, 1], tickformat='.0%')))
        return fig

    def _mc_fan(self, items, styles):
        fig = go.Figure()
        for i, (it, st) in enumerate(zip(items, styles)):
            res = it['results']
            yrs = res['record_years']
            p = {q: np.percentile(res['portfolio_values'], q, axis=0) for q in PCTS}
            vis = i == 0
            band = lambda lo, hi, alpha, name: [
                go.Scatter(x=yrs, y=p[hi], meta=str(i), visible=vis, line=dict(width=0), showlegend=False,
                           hoverinfo='skip'),
                go.Scatter(x=yrs, y=p[lo], meta=str(i), visible=vis, line=dict(width=0), fill='tonexty',
                           fillcolor=self._wash(st['pair'], alpha), name=name, hoverinfo='skip')]
            for tr in band(10, 90, 0.12, '10th–90th percentile') + band(25, 75, 0.22, '25th–75th percentile'):
                fig.add_trace(tr)
            for q, dash in ((10, 'dot'), (90, 'dot')):
                fig.add_trace(go.Scatter(x=yrs, y=p[q], meta=str(i), visible=vis, showlegend=False,
                                         name=f'{q}th', line=dict(width=0.5, color=self._c(st['pair'])),
                                         hovertemplate=f'{q}th: %{{y:$,.0f}}'))
            fig.add_trace(go.Scatter(x=yrs, y=p[50], meta=str(i), visible=vis, name='Median',
                                     line=self._line(st), hovertemplate='Median: %{y:$,.0f}'))
        invested = self._invested_curve(items)
        if invested is not None:
            fig.add_trace(go.Scatter(x=invested[0], y=invested[1], meta='all', name='Invested capital',
                                     line=dict(color=self._chrome('muted'), width=1.5, shape='hv'),
                                     hovertemplate='Invested: %{y:$,.0f}'))
        fig.update_layout(**self._layout(height=440, yfmt='$,.0f', ytitle='Balance (log scale)', ylog=True,
                                         xtitle='Years'))
        return fig

    def _invested_curve(self, items):
        sim = self.simulator
        if sim is None or not sim.contrib_amount:
            return None
        res = items[0]['results']
        dpy = res['days_per_year']
        steps = np.round(np.asarray(res['record_years']) * dpy).astype(int)
        from engine import contribution_schedule
        sched = contribution_schedule(sim.contrib_freq, int(steps[-1]), days_per_year=dpy)
        cum = np.concatenate([[0], np.cumsum(sched)])[steps]
        return res['record_years'], sim.initial_capital + sim.contrib_amount * cum

    def _mc_medians(self, items, styles):
        fig = go.Figure()
        for it, st in zip(items, styles):
            res = it['results']
            fig.add_trace(go.Scatter(x=res['record_years'], y=np.median(res['portfolio_values'], axis=0),
                                     name=it['label'], line=self._line(st), hovertemplate='%{y:$,.0f}'))
        fig.update_layout(**self._layout(yfmt='$,.0f', ytitle='Median balance (log scale)', ylog=True,
                                         xtitle='Years'))
        return fig

    def _mc_cagr_boxes(self, items, styles):
        fig = go.Figure()
        for it, st in zip(items, styles):
            pc = it['results']['stats']['percentiles']['cagr']
            fig.add_trace(go.Box(
                y=[it['label']], q1=[pc[25]], median=[pc[50]], q3=[pc[75]],
                lowerfence=[pc[10]], upperfence=[pc[90]], name=it['label'], orientation='h',
                fillcolor=self._wash(st['pair'], 0.18), line=dict(color=self._c(st['pair']), width=2),
                hoverinfo='x'))
        fig.update_layout(**self._layout(height=max(220, 44 * len(items) + 80), hover='closest', legend=False,
                                         xtitle='Annualized time-weighted return (10th–90th percentile)',
                                         xaxis=dict(self._layout()['xaxis'], tickformat='.0%'),
                                         yaxis=dict(self._layout()['yaxis'], autorange='reversed',
                                                    showgrid=False)))
        return fig

    def _prob_loss(self, items, styles):
        fig = go.Figure()
        for it, st in zip(items, styles):
            pr = it['results']['probabilities']
            fig.add_trace(go.Scatter(x=pr['years'], y=pr['prob_loss'], name=it['label'],
                                     line=self._line(st), mode='lines+markers',
                                     marker=dict(size=7, line=dict(width=2, color=self._chrome('surface'))),
                                     hovertemplate='%{y:.1%}'))
        fig.update_layout(**self._layout(yfmt='.0%', ytitle='P(balance < invested)', xtitle='Years',
                                         yaxis=dict(self._layout()['yaxis'], tickformat='.0%', rangemode='tozero')))
        return fig

    # ------------------------------------------------------------------ tables

    @staticmethod
    def _table(headers: List[str], rows: List[List[str]], first_col_html=False, cls='',
               html_cols=()) -> str:
        """Cells are escaped unless their column holds HTML built by this module."""
        raw = set(html_cols) | ({0} if first_col_html else set())
        head = ''.join(f'<th>{esc(h)}</th>' for h in headers)
        body = ''
        for row in rows:
            cells = ''.join(f'<td>{c if j in raw else esc(c)}</td>' for j, c in enumerate(row))
            body += f'<tr>{cells}</tr>'
        return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    @staticmethod
    def _name_cell(i, label, sub=None):
        sub_html = f'<span class="sub">{esc(sub)}</span>' if sub else ''
        return f'<span class="key k{i}"></span><span class="name">{esc(label)}</span>{sub_html}'

    def _summary_table(self, items):
        has_dca = any(it['backtest']['metrics']['Total Contributions'] for it in items)
        headers = ['Portfolio', 'Initial', 'Contributions', 'Final balance', 'CAGR'] + \
                  (['IRR'] if has_dca else []) + \
                  ['Stdev', 'Best year', 'Worst year', 'Max drawdown', 'Sharpe', 'Sortino', 'Calmar']
        rows = []
        for i, it in enumerate(items):
            m = it['backtest']['metrics']
            rows.append([self._name_cell(i, it['label'], it['backtest']['strategy']),
                         money(m['Start Balance']), money(m['Total Contributions']), money(m['End Balance']),
                         pct(m['CAGR'])] + ([pct(m['IRR'])] if has_dca else []) +
                        [pct(m['Stdev']), pct(m['Best Year']), pct(m['Worst Year']), pct(m['Max Drawdown']),
                         num(m['Sharpe']), num(m['Sortino']), num(m['Calmar'])])
        return self._table(headers, rows, first_col_html=True)

    def _risk_table(self, items):
        headers = ['Portfolio', 'Beta', 'Alpha', 'R²', 'Correlation', 'Upside capture', 'Downside capture',
                   'Tracking error', 'Info ratio', 'VaR 5%', 'CVaR 5%', 'Skew', 'Excess kurtosis',
                   'Positive periods', 'Rebalances', 'Costs']
        rows = []
        for i, it in enumerate(items):
            m = it['backtest']['metrics']
            rows.append([self._name_cell(i, it['label']), num(m.get('Beta')), pct(m.get('Alpha')),
                         pct(m.get('R2'), 1), num(m.get('Correlation')), pct(m.get('Upside Capture'), 1),
                         pct(m.get('Downside Capture'), 1), pct(m.get('Tracking Error')),
                         num(m.get('Info Ratio')), pct(m['VaR 5%']), pct(m['CVaR 5%']), num(m['Skewness']),
                         num(m['Excess Kurtosis']), pct(m['Positive Periods'], 1), str(m['Rebalances']),
                         money(m['Transaction Costs'])])
        return self._table(headers, rows, first_col_html=True)

    def _annual_table(self, items):
        years = sorted({y for it in items for y in it['backtest']['annual_returns'].index})
        headers = ['Year'] + [it['label'] for it in items]
        rows = []
        for y in years:
            partial = any(it['backtest']['annual_returns']['partial'].get(y, False) for it in items)
            row = [f"{y}{'†' if partial else ''}"]
            for it in items:
                a = it['backtest']['annual_returns']
                row.append(pct(a.loc[y, 'return']) if y in a.index else '—')
            rows.append(row)
        return self._table(headers, rows)

    def _drawdown_table(self, items, top=3):
        headers = ['Portfolio', 'Rank', 'Peak', 'Trough', 'Recovered', 'Depth', 'Decline (days)',
                   'Recovery (days)', 'Underwater (days)']
        rows = []
        for i, it in enumerate(items):
            for rank, d in enumerate(it['backtest']['drawdown_periods'][:top], 1):
                rows.append([self._name_cell(i, it['label']) if rank == 1 else '', str(rank), str(d['start']),
                             str(d['trough']), str(d['end']) if d['end'] else 'Not yet',
                             pct(d['depth']), f"{d['decline_days']:,}",
                             f"{d['recovery_days']:,}" if d['recovery_days'] is not None else '—',
                             f"{d['underwater_days']:,}"])
        return self._table(headers, rows, first_col_html=True)

    def _mc_table(self, items):
        has_dca = any(it['results']['stats']['total_invested'] > it['results']['portfolio_values'][0, 0]
                      for it in items)
        headers = ['Portfolio'] + [f'Balance p{q}' for q in PCTS] + ['Median CAGR', 'CAGR p10–p90'] + \
                  (['Median IRR'] if has_dca else []) + \
                  ['Median max DD', 'Max DD p5', 'Median stdev', 'P(loss)', 'P(2× invested)']
        rows = []
        for i, it in enumerate(items):
            s = it['results']['stats']
            pc = s['percentiles']
            rows.append([self._name_cell(i, it['label'])] + [money(pc['final_value'][q]) for q in PCTS] +
                        [pct(s['median_cagr']), f"{pct(pc['cagr'][10], 1)} to {pct(pc['cagr'][90], 1)}"] +
                        ([pct(s['median_irr'])] if has_dca else []) +
                        [pct(s['median_max_drawdown']), pct(s['max_drawdown_95']),
                         pct(s.get('median_volatility')), pct(s['probability_loss'], 1),
                         pct(s['probability_double'], 1)])
        return self._table(headers, rows, first_col_html=True)

    def _allocation_table(self, items, start_year, end_year):
        tickers = []
        for it in items:
            for a, w in zip(it['results']['assets'], it['results']['allocations']):
                if w > 0.001 and a['ticker'] not in tickers:
                    tickers.append(a['ticker'])
        sim = self.simulator
        initial = sim.initial_capital if sim else 10000
        years = sim.years if sim else 30
        headers = ['Portfolio'] + tickers + ['Portfolio Visualizer']
        rows = []
        for i, it in enumerate(items):
            alloc = {a['ticker']: w for a, w in zip(it['results']['assets'], it['results']['allocations'])
                     if w > 0.001}
            csv_b64 = base64.b64encode(export_portfolio_csv(alloc, it['label']).encode()).decode()
            safe = ''.join(c if c.isalnum() or c in '_-' else '_' for c in it['label'])
            links = (f'<a href="data:text/csv;base64,{csv_b64}" download="{esc(safe)}.csv">CSV</a> · '
                     f'<a href="{esc(generate_pv_url(alloc, start_year, end_year, initial))}" target="_blank" rel="noopener">Backtest</a> · '
                     f'<a href="{esc(generate_mc_url(alloc, initial, years))}" target="_blank" rel="noopener">Monte Carlo</a>')
            rows.append([self._name_cell(i, it['label'])] +
                        [pct(alloc[t], 1) if t in alloc else '—' for t in tickers] + [links])
        return self._table(headers, rows, first_col_html=True, html_cols=(len(headers) - 1,))

    def _events_section(self, items):
        blocks = []
        for i, it in enumerate(items):
            events = it['backtest']['events']
            if not events:
                continue
            tickers = list(it['backtest']['weights'].columns)
            labels = {'calendar': 'calendar rebalances', 'signal': 'signal rebalances',
                      'threshold': 'band rebalances', 'strategy': 'strategy-weighted contributions'}
            counts = pd.Series([e['trigger'] for e in events]).value_counts()
            summary = ', '.join(f"{n} {labels.get(k, k)}" for k, n in counts.items())
            shown = [e for e in events if e['trigger'] != 'calendar'][:40] or events[:12]
            rows = [[str(e['date'].date()), e['type'], e['trigger'] or '',
                     ' / '.join(f"{t} {w:.0%}" for t, w in zip(tickers, e['weights']))] for e in shown]
            blocks.append(f'<details><summary>{self._name_cell(i, it["label"])} '
                          f'<span class="sub">{esc(summary)}</span></summary>'
                          f'{self._table(["Date", "Action", "Trigger", "Target weights"], rows)}</details>')
        return ''.join(blocks)

    # ------------------------------------------------------------------ page

    def generate_html_report(self, portfolio_results, filename, start_date=None, end_date=None,
                             title: str = 'Portfolio Analysis', assumptions: Optional[Dict[str, str]] = None,
                             synthetic: bool = False, embed_plotlyjs: bool = False):
        items = [it for it in portfolio_results if it.get('backtest') and it.get('results')]
        if not items:
            raise ValueError("No results to report")
        self._cmap = {}
        styles = self._assign_styles(items)

        figs = {
            'growth': self._growth(items, styles),
            'annual': self._annual_bars(items, styles),
            'drawdowns': self._drawdowns(items, styles),
            'rolling': self._rolling(items, styles),
            'riskret': self._risk_return(items, styles),
            'monthly': self._monthly_heatmap(items),
            'allocation': self._allocation(items),
            'mcfan': self._mc_fan(items, styles),
            'mcmedian': self._mc_medians(items, styles),
            'mcbox': self._mc_cagr_boxes(items, styles),
            'mcloss': self._prob_loss(items, styles),
        }
        corr = self._correlation(items)
        if corr is not None:
            figs['corr'] = corr[0]

        # Open the allocation chart on the most interesting portfolio
        alloc_default = next((i for i, it in enumerate(items)
                              if any(e['trigger'] != 'calendar' for e in it['backtest']['events'])),
                             next((i for i, it in enumerate(items)
                                   if it['backtest']['weights'].shape[1] > 1), 0))
        start_year = start_date.year if start_date else items[0]['backtest']['dates'][0].year
        end_year = end_date.year if end_date else items[0]['backtest']['dates'][-1].year
        sim = self.simulator
        mc_units = "today's dollars" if items[0]['results'].get('real_dollars') else 'nominal dollars'

        def portfolio_options(selected=0):
            return ''.join(f'<option value="{i}"{" selected" if i == selected else ""}>{esc(it["label"])}</option>'
                           for i, it in enumerate(items))

        def selector(chart_id, opts=None, label='Portfolio'):
            opts = opts or portfolio_options()
            return (f'<label class="select">{esc(label)} <select data-chart="{chart_id}">{opts}</select></label>')

        window_opts = ''.join(f'<option value="{w}">{w}</option>' for w in ('1y', '3y', '5y'))

        def chart(chart_id, caption=''):
            cap = f'<p class="caption">{caption}</p>' if caption else ''
            return f'<div class="chart" id="c-{chart_id}"></div>{cap}'

        assumptions = assumptions or {}
        assumption_html = ''.join(f'<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>' for k, v in assumptions.items())
        banner = ('<div class="banner"><strong>Synthetic data.</strong> Prices were generated by '
                  'synthetic_data.py for offline testing. These numbers say nothing about real markets.</div>'
                  if synthetic else '')

        key_css = '\n'.join(f'.k{i}{{background:{st["pair"][0]}}}' for i, st in enumerate(styles))
        key_css_dark = '\n'.join(f'.k{i}{{background:{st["pair"][1]}}}' for i, st in enumerate(styles))

        sections = [
            ('summary', 'Summary', f'''
                <h2>Performance summary</h2>
                <p class="lede">Historical backtest over {esc(start_date)} to {esc(end_date)}. Return metrics are
                time-weighted, so contributions never count as returns. IRR is the money-weighted return
                including contributions.</p>
                {self._summary_table(items)}
                <h3>Risk and benchmark statistics</h3>
                {self._risk_table(items)}
                <details><summary>Allocations and Portfolio Visualizer export</summary>
                {self._allocation_table(items, start_year, end_year)}</details>'''),
            ('growth', 'Growth', f'''
                <h2>Portfolio growth</h2>{chart('growth')}'''),
            ('returns', 'Returns', f'''
                <h2>Annual returns</h2>{chart('annual', '† partial calendar year (measured from the first or to the last available date).')}
                <details><summary>Annual returns table</summary>{self._annual_table(items)}</details>
                <h2>Monthly returns</h2>{selector('monthly')}{chart('monthly')}
                <h2>Rolling returns</h2>{selector('rolling', window_opts, 'Window')}{chart('rolling')}'''),
            ('drawdowns', 'Drawdowns', f'''
                <h2>Drawdowns</h2>{chart('drawdowns')}
                <h3>Largest drawdowns</h3>{self._drawdown_table(items)}'''),
            ('risk', 'Risk', f'''
                <h2>Risk vs return</h2>{chart('riskret', 'Annualized stdev of monthly returns vs CAGR, historical.')}
                {('<h2>Asset correlations</h2>' + chart('corr', 'Correlation of monthly returns over each pair’s overlapping history.')) if corr is not None else ''}'''),
            ('allocation', 'Allocation', f'''
                <h2>Allocation over time</h2>{selector('allocation', portfolio_options(alloc_default))}
                {chart('allocation', 'Weekly snapshot of holdings weights. Vertical lines mark signal- or band-triggered rebalances.')}
                {('<h3>Strategy and rebalance events</h3>' + self._events_section(items)) if any(it['backtest']['events'] for it in items) else ''}'''),
            ('montecarlo', 'Monte Carlo', f'''
                <h2>Monte Carlo simulation</h2>
                <p class="lede">{esc(sim.simulations if sim else '')} paths over {esc(sim.years if sim else '')} years,
                method <code>{esc(items[0]['results'].get('method'))}</code>, history
                {esc(items[0]['results'].get('history_start'))} to {esc(items[0]['results'].get('history_end'))}.
                Balances in {mc_units}.</p>
                {self._mc_table(items)}
                <h3>Range of outcomes</h3>{selector('mcfan')}{chart('mcfan')}
                <h3>Median balance</h3>{chart('mcmedian')}
                <h3>Annualized return distribution</h3>{chart('mcbox', 'Box = 25th–75th percentile, whiskers = 10th–90th, line = median.')}
                <h3>Probability of loss</h3>{chart('mcloss', 'Share of paths whose balance is below the money invested so far.')}'''),
            ('notes', 'Notes', f'''
                <h2>Assumptions and methodology</h2>
                <dl class="assumptions">{assumption_html}</dl>
                <ul class="notes">
                  <li><b>CAGR</b> is time-weighted and annualized by calendar span, so 24/7 crypto and 5-day equity calendars compare correctly.</li>
                  <li><b>Stdev, Sharpe, Sortino, beta, capture ratios, VaR</b> use monthly returns (Portfolio Visualizer's convention); histories under a year fall back to daily. Sharpe and Sortino subtract the configured risk-free rate.</li>
                  <li><b>Max drawdown</b> uses daily values, so it can be deeper than Portfolio Visualizer's month-end figure.</li>
                  <li><b>Best/Worst year</b> are calendar-year returns over full years.</li>
                  <li><b>Upside/Downside capture</b> are ratios (1.00 = matches the benchmark).</li>
                  <li><b>Optimized portfolios</b> are fit to the same history they are tested on (in-sample); expect worse results going forward.</li>
                  <li><b>Monte Carlo</b> resamples or fits the common history of each portfolio's assets; it cannot produce regimes that history doesn't contain.</li>
                </ul>'''),
        ]
        nav = ''.join(f'<a href="#{sid}">{esc(name)}</a>' for sid, name, _ in sections)
        body = ''.join(f'<section id="{sid}">{content}</section>' for sid, _, content in sections)

        fig_payload = json.dumps({f'c-{k}': self._fig_json(v) for k, v in figs.items()})
        cmap = json.dumps(self._cmap)
        page = _PAGE.format(
            title=esc(title), period=f"{esc(start_date)} to {esc(end_date)}", banner=banner, nav=nav, body=body,
            figs=fig_payload, cmap=cmap,
            plotly_script=(f'<script>{get_plotlyjs()}</script>' if embed_plotlyjs else
                           f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>'),
            key_css=key_css, key_css_dark=key_css_dark,
            **{f'{k}_l': v[0] for k, v in CHROME.items()}, **{f'{k}_d': v[1] for k, v in CHROME.items()},
        )
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(page)


_PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{plotly_script}
<style>
:root {{
  color-scheme: light;
  --page: {page_l}; --surface: {surface_l}; --text: {text_l}; --text2: {text2_l};
  --muted: {muted_l}; --grid: {grid_l}; --axis: {axis_l};
  --border: rgba(11,11,11,0.10);
}}
{key_css}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --page: {page_d}; --surface: {surface_d}; --text: {text_d}; --text2: {text2_d};
    --muted: {muted_d}; --grid: {grid_d}; --axis: {axis_d}; --border: rgba(255,255,255,0.10);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: {page_d}; --surface: {surface_d}; --text: {text_d}; --text2: {text2_d};
  --muted: {muted_d}; --grid: {grid_d}; --axis: {axis_d}; --border: rgba(255,255,255,0.10);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--text); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
header {{ max-width: 1240px; margin: 0 auto; padding: 24px 16px 8px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }}
header h1 {{ font-size: 22px; margin: 0; font-weight: 600; }}
header .period {{ color: var(--text2); }}
header button {{ margin-left: auto; background: var(--surface); color: var(--text2); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font: inherit; cursor: pointer; }}
nav {{ position: sticky; top: 0; z-index: 5; background: var(--page); border-bottom: 1px solid var(--border); }}
nav div {{ max-width: 1240px; margin: 0 auto; padding: 0 16px; display: flex; gap: 4px; overflow-x: auto; }}
nav a {{ color: var(--text2); text-decoration: none; padding: 10px 10px; white-space: nowrap; border-bottom: 2px solid transparent; }}
nav a:hover {{ color: var(--text); border-bottom-color: var(--axis); }}
main {{ max-width: 1240px; margin: 0 auto; padding: 8px 16px 48px; }}
section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px 20px; margin: 16px 0; scroll-margin-top: 52px; }}
h2 {{ font-size: 17px; font-weight: 600; margin: 8px 0 8px; }}
h3 {{ font-size: 14px; font-weight: 600; margin: 20px 0 6px; color: var(--text2); }}
.lede {{ color: var(--text2); margin: 0 0 12px; max-width: 80ch; }}
.caption {{ color: var(--muted); font-size: 12px; margin: 4px 0 0; }}
.banner {{ max-width: 1208px; margin: 8px auto 0; padding: 10px 14px; border-radius: 8px; border: 1px solid #fab219; background: rgba(250,178,25,0.12); }}
.table-wrap {{ overflow-x: auto; margin: 4px 0 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; font-variant-numeric: tabular-nums; }}
th {{ color: var(--text2); font-weight: 600; font-size: 12px; position: sticky; top: 0; background: var(--surface); }}
th:first-child, td:first-child {{ text-align: left; }}
td:first-child {{ white-space: normal; min-width: 180px; }}
.key {{ display: inline-block; width: 12px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 8px; }}
.name {{ font-weight: 600; }}
.sub {{ display: block; color: var(--muted); font-size: 12px; font-weight: 400; margin-left: 20px; }}
details {{ margin: 12px 0; }}
summary {{ cursor: pointer; color: var(--text2); font-weight: 600; }}
details summary .sub {{ display: inline; }}
a {{ color: var(--text); }}
code {{ font-size: 12px; }}
.chart {{ width: 100%; min-height: 220px; }}
.select {{ display: inline-flex; gap: 8px; align-items: center; color: var(--text2); margin: 4px 0 8px; }}
.select select {{ font: inherit; color: var(--text); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; max-width: 70vw; }}
dl.assumptions {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px 24px; margin: 8px 0 16px; }}
dl.assumptions dt {{ color: var(--muted); font-size: 12px; }}
dl.assumptions dd {{ margin: 0; }}
ul.notes {{ color: var(--text2); padding-left: 18px; max-width: 90ch; }}
</style>
<style id="dark-keys" media="not all">{key_css_dark}</style>
</head>
<body>
<header><h1>{title}</h1><span class="period">{period}</span><button id="theme" type="button">Theme: auto</button></header>
{banner}
<nav><div>{nav}</div></nav>
<main>{body}</main>
<script>
const FIGS = {figs};
const CMAP = {cmap};
const selection = {{}};
const root = document.documentElement;
const mq = window.matchMedia('(prefers-color-scheme: dark)');

function isDark() {{
  const t = root.dataset.theme;
  return t === 'dark' || (t !== 'light' && mq.matches);
}}
function swap(o) {{
  if (typeof o === 'string') return CMAP[o] ?? o;
  if (Array.isArray(o)) return o.map(swap);
  if (o && typeof o === 'object') {{ const r = {{}}; for (const k in o) r[k] = swap(o[k]); return r; }}
  return o;
}}
function selectionFor(id) {{
  const el = document.querySelector('select[data-chart="' + id.slice(2) + '"]');
  return el ? el.value : null;
}}
function applySelection(fig, sel) {{
  if (sel === null) return fig;
  fig.data.forEach(t => {{ if (t.meta !== undefined && t.meta !== null) t.visible = (t.meta === 'all' || String(t.meta) === sel); }});
  (fig.layout.shapes || []).forEach(s => {{ if (s.name !== undefined) s.visible = String(s.name) === sel; }});
  return fig;
}}
function render() {{
  const dark = isDark();
  document.getElementById('dark-keys').media = dark ? 'all' : 'not all';
  for (const id in FIGS) {{
    const el = document.getElementById(id);
    if (!el) continue;
    let fig = JSON.parse(JSON.stringify(FIGS[id]));
    if (dark) fig = swap(fig);
    fig = applySelection(fig, selectionFor(id));
    Plotly.react(el, fig.data, fig.layout, {{responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d']}});
  }}
}}
document.querySelectorAll('select[data-chart]').forEach(s => s.addEventListener('change', render));
const modes = ['auto', 'light', 'dark'];
document.getElementById('theme').addEventListener('click', e => {{
  const next = modes[(modes.indexOf(root.dataset.theme || 'auto') + 1) % 3];
  if (next === 'auto') delete root.dataset.theme; else root.dataset.theme = next;
  e.target.textContent = 'Theme: ' + next;
  render();
}});
mq.addEventListener('change', render);
render();
</script>
</body>
</html>
'''
