"""
Generates figures/summary_metrics.png — the single chart at the top of the README.

Every value is parsed straight out of results.md, the committed raw stdout of the
experiment scripts, so the figure cannot drift from the recorded results. No model
is re-run here.

Usage:
    python code/fig_summary.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE_DIR, 'results.md')
OUT = os.path.join(BASE_DIR, 'figures', 'summary_metrics.png')

TICKERS = ['GOOGL', 'MSFT', 'NVDA', 'TSLA', 'NDX']

BLOCKS = ('group1A', 'group1B', 'group2A', 'group2B', 'group3')

# what to pull from results.md, in bar order. 'table' selects which table inside
# the block (group2B holds the 30-day run first, then the 200-day run); None takes
# the last one.
SERIES = [
    {'label': 'Static (full history)', 'short': 'Static',
     'block': 'group1B', 'table': None, 'color': '#b0b0b0'},
    {'label': 'Sliding 30d', 'short': '30d',
     'block': 'group2B', 'table': 1, 'color': '#9ecae1'},
    {'label': 'Sliding 200d', 'short': '200d',
     'block': 'group2B', 'table': 2, 'color': '#3182bd'},
    {'label': 'Adaptive 15-200d', 'short': 'Adaptive',
     'block': 'group3', 'table': None, 'color': '#e6550d'},
]

ROW = re.compile(
    r'^([A-Z]{2,5})\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)'
)


def parse_results():
    """{block: {table_index: {ticker: {'RMSE','MAPE','DA','Regret'}}}}, read from results.md."""
    text = open(RESULTS, encoding='utf-8').read()

    blocks, current = {}, None
    for line in text.splitlines():
        marker = line.strip().strip('`').strip()
        if marker in BLOCKS:
            current = marker
            blocks[current] = {0: {}}
            continue
        if current is None:
            continue
        if line.strip().startswith('TABLE '):
            # a second (or third) table inside the same block
            blocks[current][max(blocks[current]) + 1] = {}
            continue
        m = ROW.match(line.strip())
        if m and m.group(1) in TICKERS:
            ticker, rmse, mape, da, regret, _t = m.groups()
            blocks[current][max(blocks[current])][ticker] = {
                'RMSE': float(rmse), 'MAPE': float(mape),
                'DA': float(da), 'Regret': float(regret),
            }

    data = {}
    for cfg in SERIES:
        tables = blocks[cfg['block']]
        idx = max(tables) if cfg['table'] is None else cfg['table']
        picked = tables[idx]
        if len(picked) != len(TICKERS):
            raise SystemExit(
                f"{cfg['label']}: expected {len(TICKERS)} tickers from "
                f"{cfg['block']}[{idx}], got {sorted(picked)}"
            )
        data[cfg['label']] = picked
    return data


def grouped_bars(ax, data, metric, title, ylabel):
    n, w = len(TICKERS), 0.2
    x = np.arange(n)
    top = 0

    for i, cfg in enumerate(SERIES):
        vals = [data[cfg['label']][t][metric] for t in TICKERS]
        top = max(top, max(vals))
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=cfg['label'], color=cfg['color'],
                      edgecolor='white', linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, top * 1.30)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def annotate_winners(ax, data, metric, better):
    """Label each ticker with the configuration(s) attaining the best value.

    Ties are printed as ties. Several DA columns here coincide exactly, and naming
    only the first of them would overstate the adaptive model's lead.
    """
    for i, t in enumerate(TICKERS):
        vals = [data[cfg['label']][t][metric] for cfg in SERIES]
        best = better(vals)
        names = [cfg['short'] for cfg, v in zip(SERIES, vals) if v == best]
        ax.text(i, ax.get_ylim()[1] * 0.985, ' / '.join(names),
                ha='center', va='top', fontsize=7.0, style='italic', color='#444444')


def main():
    data = parse_results()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=200)

    grouped_bars(axes[0], data, 'Regret',
                 'Return forgone vs a perfect-foresight strategy  (lower is better)',
                 'Cumulative Regret (%)')
    grouped_bars(axes[1], data, 'DA',
                 'Direction accuracy  (higher is better)',
                 'DA (%)')

    annotate_winners(axes[0], data, 'Regret', min)
    annotate_winners(axes[1], data, 'DA', max)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        'Regime shift — trained on the 2021 bull market, tested on the early-2022 crash '
        '(same ~40-day window for all four configurations)',
        fontsize=11.5, y=0.98,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.93])
    fig.savefig(OUT, bbox_inches='tight', facecolor='white')
    print(f'wrote {os.path.relpath(OUT, BASE_DIR)}')

    # print what was parsed, so the numbers on the chart can be eyeballed against results.md
    for cfg in SERIES:
        print(f"  {cfg['label']:22s}", ' '.join(
            f'{t}={data[cfg["label"]][t]["Regret"]:.2f}/{data[cfg["label"]][t]["DA"]:.1f}'
            for t in TICKERS))


if __name__ == '__main__':
    main()
