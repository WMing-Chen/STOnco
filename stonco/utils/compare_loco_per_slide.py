#!/usr/bin/env python3
"""
Compare multiple LOCO per-slide result CSVs with boxplots and significance tests.

Examples:
python -m stonco.utils.compare_loco_per_slide \
  --csv /path/to/run_a/loco_per_slide.csv \
  --csv /path/to/run_b/loco_per_slide.csv \
  --label no_domain \
  --label with_domain \
  --metrics accuracy

python -m stonco.utils.compare_loco_per_slide \
  --csv /path/to/run1.csv \
  --csv /path/to/run2.csv \
  --csv /path/to/run3.csv \
  --csv /path/to/run4.csv \
  --label baseline \
  --label da \
  --label mmd \
  --label da_mmd \
  --metrics auroc,auprc,accuracy,macro_f1
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError('scipy is required for compare_loco_per_slide.py') from exc


DEFAULT_METRICS = ['accuracy']
KNOWN_METRICS = ['auroc', 'auprc', 'accuracy', 'macro_f1', 'pos_rate']
KEY_COLS = ['cancer_type', 'slide_id']


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple loco_per_slide.csv files with boxplots and significance tests.')
    parser.add_argument('--csv', action='append', default=None, help='Can be specified multiple times for multi-group comparison.')
    parser.add_argument('--label', action='append', default=None, help='Can be specified multiple times; order must match --csv.')
    parser.add_argument('--csv_a', default=None, help='Backward-compatible first loco_per_slide.csv path.')
    parser.add_argument('--csv_b', default=None, help='Backward-compatible second loco_per_slide.csv path.')
    parser.add_argument('--label_a', default='Run A', help='Backward-compatible label for the first run.')
    parser.add_argument('--label_b', default='Run B', help='Backward-compatible label for the second run.')
    parser.add_argument('--metrics', default=','.join(DEFAULT_METRICS), help='Comma-separated metrics to compare.')
    parser.add_argument('--out_dir', default=None, help='Output directory.')
    parser.add_argument(
        '--test',
        choices=['auto', 'paired', 'welch', 'friedman', 'kruskal'],
        default='auto',
        help='Global significance test. auto: paired_t/ Friedman for matched groups, otherwise Welch/Kruskal.',
    )
    parser.add_argument('--formats', default='svg,png', help='Comma-separated output figure formats.')
    parser.add_argument('--dpi', type=int, default=180, help='DPI for raster outputs.')
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(text).strip()).strip('_') or 'run'


def parse_runs(args):
    if args.csv:
        csvs = list(args.csv)
        labels = list(args.label or [])
        if labels and len(labels) != len(csvs):
            raise ValueError(f'Number of --label ({len(labels)}) must match number of --csv ({len(csvs)}).')
        if not labels:
            labels = [f'Run{i+1}' for i in range(len(csvs))]
        if len(csvs) < 2:
            raise ValueError('Please provide at least two --csv inputs.')
        return [{'path': p, 'label': l, 'safe_label': safe_name(l)} for p, l in zip(csvs, labels)]

    if args.csv_a and args.csv_b:
        return [
            {'path': args.csv_a, 'label': args.label_a, 'safe_label': safe_name(args.label_a)},
            {'path': args.csv_b, 'label': args.label_b, 'safe_label': safe_name(args.label_b)},
        ]

    raise ValueError('Please specify either repeated --csv/--label or the backward-compatible --csv_a/--csv_b pair.')


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'{path} missing required columns: {missing}. Found: {list(df.columns)}')
    return df


def resolve_metrics(dfs, metrics_raw: str):
    metrics = [m.strip() for m in str(metrics_raw).split(',') if m.strip()]
    if not metrics:
        metrics = list(DEFAULT_METRICS)
    unknown = [m for m in metrics if m not in KNOWN_METRICS]
    if unknown:
        raise ValueError(f'Unknown metrics: {unknown}. Supported: {KNOWN_METRICS}')
    missing = [m for m in metrics if any(m not in df.columns for df in dfs)]
    if missing:
        raise ValueError(f'Metrics missing from at least one input CSV: {sorted(set(missing))}')
    return metrics


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return 'n/a'
    if p < 1e-4:
        return '****'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full_like(pvalues, np.nan, dtype=float)
    finite_mask = np.isfinite(pvalues)
    finite_vals = pvalues[finite_mask]
    if finite_vals.size == 0:
        return out
    order = np.argsort(finite_vals)
    ranked = finite_vals[order]
    n = float(len(ranked))
    adj = np.empty_like(ranked)
    prev = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        rank = i + 1.0
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    out_idx = np.where(finite_mask)[0]
    out[out_idx] = adj[inv_order]
    return out


def save_figure(fig, out_base: Path, formats, dpi: int):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        out_path = out_base.with_suffix('.' + fmt)
        if fmt in {'png', 'jpg', 'jpeg'}:
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(out_path, bbox_inches='tight')


def build_metric_frame(runs, metric: str):
    merged = None
    value_cols = []
    for idx, run in enumerate(runs):
        value_col = f'value_{idx}'
        sub = run['df'][KEY_COLS + [metric]].copy().rename(columns={metric: value_col})
        merged = sub if merged is None else merged.merge(sub, on=KEY_COLS, how='inner')
        value_cols.append(value_col)
    if merged is None:
        return pd.DataFrame(), []
    merged = merged.dropna(subset=value_cols)
    return merged, value_cols


def normalize_arrays(values_list):
    out = []
    for vals in values_list:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        out.append(vals)
    return out


def run_global_test(values_list, test_mode: str, matched: bool):
    values_list = normalize_arrays(values_list)
    if len(values_list) < 2 or any(len(v) < 2 for v in values_list):
        return {'test_name': 'insufficient', 'statistic': float('nan'), 'pvalue': float('nan')}

    if test_mode == 'paired':
        if len(values_list) == 2:
            stat, pvalue = stats.ttest_rel(values_list[0], values_list[1], nan_policy='omit')
            return {'test_name': 'paired_t', 'statistic': float(stat), 'pvalue': float(pvalue)}
        stat, pvalue = stats.friedmanchisquare(*values_list)
        return {'test_name': 'friedman', 'statistic': float(stat), 'pvalue': float(pvalue)}
    if test_mode == 'welch':
        if len(values_list) == 2:
            stat, pvalue = stats.ttest_ind(values_list[0], values_list[1], equal_var=False, nan_policy='omit')
            return {'test_name': 'welch_t', 'statistic': float(stat), 'pvalue': float(pvalue)}
        stat, pvalue = stats.kruskal(*values_list)
        return {'test_name': 'kruskal', 'statistic': float(stat), 'pvalue': float(pvalue)}
    if test_mode == 'friedman':
        stat, pvalue = stats.friedmanchisquare(*values_list)
        return {'test_name': 'friedman', 'statistic': float(stat), 'pvalue': float(pvalue)}
    if test_mode == 'kruskal':
        stat, pvalue = stats.kruskal(*values_list)
        return {'test_name': 'kruskal', 'statistic': float(stat), 'pvalue': float(pvalue)}

    if len(values_list) == 2:
        if matched:
            stat, pvalue = stats.ttest_rel(values_list[0], values_list[1], nan_policy='omit')
            return {'test_name': 'paired_t', 'statistic': float(stat), 'pvalue': float(pvalue)}
        stat, pvalue = stats.ttest_ind(values_list[0], values_list[1], equal_var=False, nan_policy='omit')
        return {'test_name': 'welch_t', 'statistic': float(stat), 'pvalue': float(pvalue)}

    if matched:
        stat, pvalue = stats.friedmanchisquare(*values_list)
        return {'test_name': 'friedman', 'statistic': float(stat), 'pvalue': float(pvalue)}
    stat, pvalue = stats.kruskal(*values_list)
    return {'test_name': 'kruskal', 'statistic': float(stat), 'pvalue': float(pvalue)}


def run_pairwise_test(vals_a, vals_b, matched: bool, n_groups: int):
    vals_a, vals_b = normalize_arrays([vals_a, vals_b])
    if len(vals_a) < 2 or len(vals_b) < 2:
        return {'test_name': 'insufficient', 'statistic': float('nan'), 'pvalue': float('nan')}

    if n_groups <= 2:
        if matched:
            stat, pvalue = stats.ttest_rel(vals_a, vals_b, nan_policy='omit')
            return {'test_name': 'paired_t', 'statistic': float(stat), 'pvalue': float(pvalue)}
        stat, pvalue = stats.ttest_ind(vals_a, vals_b, equal_var=False, nan_policy='omit')
        return {'test_name': 'welch_t', 'statistic': float(stat), 'pvalue': float(pvalue)}

    if matched:
        stat, pvalue = stats.wilcoxon(vals_a, vals_b, zero_method='wilcox', alternative='two-sided')
        return {'test_name': 'wilcoxon', 'statistic': float(stat), 'pvalue': float(pvalue)}
    stat, pvalue = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
    return {'test_name': 'mannwhitney', 'statistic': float(stat), 'pvalue': float(pvalue)}


def make_overall_row(metric, scope, cancer_type, values_list, runs, test_result, n_matched):
    row = {
        'metric': metric,
        'scope': scope,
        'cancer_type': cancer_type,
        'n_matched': int(n_matched),
        'global_test_name': test_result['test_name'],
        'global_statistic': test_result['statistic'],
        'global_pvalue': test_result['pvalue'],
        'global_significance': significance_stars(test_result['pvalue']),
    }
    for vals, run in zip(values_list, runs):
        row[f"mean_{run['safe_label']}"] = float(np.mean(vals)) if len(vals) else float('nan')
        row[f"std_{run['safe_label']}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan')
    return row


def build_pairwise_rows(metric, scope, cancer_type, values_list, runs, matched):
    rows = []
    raw_p = []
    temp = []
    n_groups = len(values_list)
    for i, j in itertools.combinations(range(n_groups), 2):
        res = run_pairwise_test(values_list[i], values_list[j], matched=matched, n_groups=n_groups)
        row = {
            'metric': metric,
            'scope': scope,
            'cancer_type': cancer_type,
            'label_i': runs[i]['label'],
            'label_j': runs[j]['label'],
            'test_name': res['test_name'],
            'statistic': res['statistic'],
            'pvalue_raw': res['pvalue'],
        }
        temp.append(row)
        raw_p.append(res['pvalue'])
    adj = bh_adjust(raw_p)
    for row, p_adj in zip(temp, adj):
        row['pvalue_bh'] = float(p_adj) if np.isfinite(p_adj) else float('nan')
        row['significance_bh'] = significance_stars(p_adj)
        rows.append(row)
    return rows


def annotate_pairwise_overall(ax, pairwise_rows, runs, positions, ymin, ymax):
    label_to_pos = {run['label']: pos for run, pos in zip(runs, positions)}
    sig_rows = [row for row in pairwise_rows if row.get('significance_bh', 'ns') not in {'ns', 'n/a'}]
    if not sig_rows:
        return ymax
    height = max(0.025, (ymax - ymin) * 0.08 if np.isfinite(ymax - ymin) else 0.05)
    top = ymax + height * 0.55
    for level, row in enumerate(sig_rows):
        x1 = label_to_pos[row['label_i']]
        x2 = label_to_pos[row['label_j']]
        if x1 > x2:
            x1, x2 = x2, x1
        y = top + level * height
        ax.plot([x1, x1, x2, x2], [y - height * 0.25, y, y, y - height * 0.25], color='black', linewidth=0.9)
        ax.text((x1 + x2) / 2, y + height * 0.04, row['significance_bh'], ha='center', va='bottom', fontsize=9)
    return top + len(sig_rows) * height


def annotate_pairwise_per_cancer(ax, pairwise_rows, runs, cancer, cancer_center, offsets, ymin, ymax):
    label_to_offset = {run['label']: off for run, off in zip(runs, offsets)}
    sig_rows = [
        row for row in pairwise_rows
        if row.get('cancer_type') == cancer and row.get('significance_bh', 'ns') not in {'ns', 'n/a'}
    ]
    if not sig_rows:
        return ymax
    height = max(0.02, (ymax - ymin) * 0.10 if np.isfinite(ymax - ymin) else 0.04)
    top = ymax + height * 0.55
    for level, row in enumerate(sig_rows):
        x1 = cancer_center + label_to_offset[row['label_i']]
        x2 = cancer_center + label_to_offset[row['label_j']]
        if x1 > x2:
            x1, x2 = x2, x1
        y = top + level * height
        ax.plot([x1, x1, x2, x2], [y - height * 0.22, y, y, y - height * 0.22], color='black', linewidth=0.75)
        ax.text((x1 + x2) / 2, y + height * 0.03, row['significance_bh'], ha='center', va='bottom', fontsize=7.5)
    return top + len(sig_rows) * height


def get_run_colors(n_runs: int):
    cmap = plt.cm.get_cmap('tab10', max(n_runs, 3))
    return [cmap(i) for i in range(n_runs)]


def plot_metric(ax, merged: pd.DataFrame, value_cols, metric: str, runs, test_mode: str):
    values_list = [merged[col].astype(float).to_numpy() for col in value_cols]
    test_result = run_global_test(values_list, test_mode, matched=True)
    colors = get_run_colors(len(runs))
    positions = np.arange(1, len(runs) + 1)
    bp = ax.boxplot(values_list, positions=positions, widths=0.55, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor('black')
    for part in ('medians', 'caps', 'whiskers'):
        for artist in bp[part]:
            artist.set_color('black')

    rng = np.random.default_rng(42)
    for pos, vals, color in zip(positions, values_list, colors):
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=20, color=color, alpha=0.85,
                   edgecolors='black', linewidths=0.25, zorder=3)

    all_vals = np.concatenate(values_list) if values_list else np.array([0.0, 1.0])
    ymax = np.nanmax(all_vals)
    ymin = np.nanmin(all_vals)
    pad = max(0.03, (ymax - ymin) * 0.16 if np.isfinite(ymax - ymin) else 0.05)
    top_y = ymax + pad * 0.35
    ax.text(np.mean([positions[0], positions[-1]]), ymax + pad * 0.35,
            f"{test_result['test_name']}: p={test_result['pvalue']:.3g} ({significance_stars(test_result['pvalue'])})",
            ha='center', va='bottom', fontsize=9)

    summary_row = make_overall_row(metric, 'overall', 'ALL', values_list, runs, test_result, len(merged))
    pairwise_rows = build_pairwise_rows(metric, 'overall', 'ALL', values_list, runs, matched=True)
    top_y = annotate_pairwise_overall(ax, pairwise_rows, runs, positions, ymin, top_y)
    ax.set_ylim(ymin - pad * 0.12, top_y + pad * 0.45)

    ax.set_xticks(positions)
    ax.set_xticklabels([run['label'] for run in runs], rotation=15, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} (matched slides: n={len(merged)})')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    return summary_row, pairwise_rows


def plot_per_cancer_metric(fig, ax, merged: pd.DataFrame, value_cols, metric: str, runs, test_mode: str):
    cancers = sorted(merged['cancer_type'].dropna().astype(str).unique().tolist())
    if not cancers:
        ax.axis('off')
        ax.set_title(f'{metric}: no matched cancers')
        return [], []

    colors = get_run_colors(len(runs))
    width = min(0.72 / max(len(runs), 1), 0.24)
    offsets = np.linspace(-(len(runs) - 1) / 2.0, (len(runs) - 1) / 2.0, len(runs)) * width
    rng = np.random.default_rng(42)
    summary_rows = []
    pairwise_rows = []
    ymax_all = []
    ymin_all = []

    for i, cancer in enumerate(cancers, start=1):
        sub = merged[merged['cancer_type'].astype(str) == cancer].copy()
        values_list = [sub[col].astype(float).to_numpy() for col in value_cols]
        if any(len(v) == 0 for v in values_list):
            continue

        for pos_offset, vals, color in zip(offsets, values_list, colors):
            pos = i + pos_offset
            bp = ax.boxplot([vals], positions=[pos], widths=width * 0.90, patch_artist=True, manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
                patch.set_edgecolor('black')
            for part in ('medians', 'caps', 'whiskers'):
                for artist in bp[part]:
                    artist.set_color('black')
            jitter = rng.uniform(-width * 0.18, width * 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=16, color=color, alpha=0.80,
                       edgecolors='black', linewidths=0.20, zorder=3)

        test_result = run_global_test(values_list, test_mode, matched=True)
        cancer_vals = np.concatenate(values_list)
        ymax = float(np.nanmax(cancer_vals))
        ymin = float(np.nanmin(cancer_vals))
        ymax_all.append(ymax)
        ymin_all.append(ymin)
        pad = max(0.02, (ymax - ymin) * 0.12 if np.isfinite(ymax - ymin) else 0.04)
        left = i + offsets[0]
        right = i + offsets[-1]
        ax.plot([left, left, right, right], [ymax + pad * 0.2, ymax + pad * 0.5, ymax + pad * 0.5, ymax + pad * 0.2],
                color='black', linewidth=0.9)
        ax.text(i, ymax + pad * 0.55, significance_stars(test_result['pvalue']), ha='center', va='bottom', fontsize=9)

        summary_rows.append(make_overall_row(metric, 'per_cancer', cancer, values_list, runs, test_result, len(sub)))
        pairwise_rows.extend(build_pairwise_rows(metric, 'per_cancer', cancer, values_list, runs, matched=True))
        ymax_all.append(annotate_pairwise_per_cancer(ax, pairwise_rows, runs, cancer, i, offsets, ymin, ymax + pad * 0.55))

    if ymax_all and ymin_all:
        ymin = min(ymin_all)
        ymax = max(ymax_all)
        pad = max(0.03, (ymax - ymin) * 0.22 if np.isfinite(ymax - ymin) else 0.05)
        ax.set_ylim(ymin - pad * 0.15, ymax + pad)

    ax.set_xticks(np.arange(1, len(cancers) + 1))
    ax.set_xticklabels(cancers, rotation=35, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by cancer type')
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markeredgecolor='black',
                   markersize=9, alpha=0.7, label=run['label'])
        for run, color in zip(runs, colors)
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, ncol=min(2, len(runs)))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    return summary_rows, pairwise_rows


def main():
    args = parse_args()
    runs = parse_runs(args)
    for run in runs:
        run['df'] = load_csv(run['path'])
    metrics = resolve_metrics([run['df'] for run in runs], args.metrics)
    formats = [x.strip() for x in str(args.formats).split(',') if x.strip()]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        safe_join = '_vs_'.join(run['safe_label'] for run in runs)
        out_dir = Path(runs[0]['path']).resolve().parent / f'compare_{safe_join}'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_rows_per_cancer = []
    pairwise_rows = []
    pairwise_rows_per_cancer = []

    ncols = min(2, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.4 * ncols, 5.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics):
        merged, value_cols = build_metric_frame(runs, metric)
        if merged.empty:
            ax.axis('off')
            ax.set_title(f'{metric}: no matched slides')
            continue
        summary_row, pair_rows = plot_metric(ax, merged, value_cols, metric, runs, args.test)
        summary_rows.append(summary_row)
        pairwise_rows.extend(pair_rows)

        fig_pc, ax_pc = plt.subplots(figsize=(max(10, len(merged['cancer_type'].unique()) * 1.0), 7.6))
        rows_pc, pair_rows_pc = plot_per_cancer_metric(fig_pc, ax_pc, merged, value_cols, metric, runs, args.test)
        summary_rows_per_cancer.extend(rows_pc)
        pairwise_rows_per_cancer.extend(pair_rows_pc)
        fig_pc.suptitle(f'Per-cancer comparison: {metric}', fontsize=13, y=0.975)
        save_figure(fig_pc, out_dir / f'compare_per_cancer_{metric}', formats, args.dpi)
        plt.close(fig_pc)

    for ax in axes[len(metrics):]:
        ax.axis('off')

    fig.suptitle('LOCO per-slide comparison', fontsize=14, y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.subplots_adjust(hspace=0.48, wspace=0.26)
    save_figure(fig, out_dir / 'compare_boxplots', formats, args.dpi)
    plt.close(fig)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / 'compare_stats.csv', index=False)
        print(f'[OK] Saved summary stats to {out_dir / "compare_stats.csv"}')
    if summary_rows_per_cancer:
        pd.DataFrame(summary_rows_per_cancer).to_csv(out_dir / 'compare_stats_per_cancer.csv', index=False)
        print(f'[OK] Saved per-cancer stats to {out_dir / "compare_stats_per_cancer.csv"}')

    print(f'[OK] Saved figures to {out_dir}')


if __name__ == '__main__':
    main()
