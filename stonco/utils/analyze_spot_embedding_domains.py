#!/usr/bin/env python3
"""
Analyze spot-level embedding distributions from an exported embeddings CSV.

This script consolidates the ad hoc analyses used for `spot_embeddings_h.csv`:
- compute per-domain mean/variance statistics for embedding columns
- plot domain overview heatmaps + boxplots
- compare selected domain-mean dimensions with smoothed density curves
- plot spot-level KDE by cancer_type for selected dimensions
- plot spot-level strip/scatter distributions by cancer_type for selected dimensions
- plot 2D spot-level scatter on two embedding dimensions, with optional balanced per-group sampling

Example:
python stonco/utils/analyze_spot_embedding_domains.py \
  --embeddings_csv test/.../embedding/spot_embeddings_h.csv \
  --group_cols cancer_type batch_id \
  --selected_dims 7 20 \
  --joint_scatter_group_col cancer_type \
  --joint_scatter_sample_caps 3000 1000
"""

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


DEFAULT_GROUP_COLS = ['cancer_type', 'batch_id']
DEFAULT_SELECTED_DIMS = [7, 20]
DEFAULT_JOINT_SCATTER_SAMPLE_CAPS = [3000, 1000]


def _sanitize_name(text: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text).strip())
    return safe.strip('_') or 'value'


def _embedding_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f'No embedding columns found with prefix: {prefix}')

    def sort_key(col: str):
        suffix = col[len(prefix):]
        try:
            return (0, int(suffix))
        except Exception:
            return (1, suffix)

    return sorted(cols, key=sort_key)


def _sorted_group_labels(values: Iterable[object], group_col: str) -> List[str]:
    labels = [str(v) for v in values]

    def sort_key(label: str):
        if group_col == 'batch_id':
            try:
                return (0, int(label))
            except Exception:
                return (1, label)
        m = re.match(r'([A-Za-z]+)(\d+)$', label)
        if m:
            return (0, m.group(1), int(m.group(2)))
        return (1, label)

    return sorted(labels, key=sort_key)


def _categorical_palette(n: int):
    cmaps = ['tab20', 'tab20b', 'tab20c', 'Set3']
    colors = []
    for name in cmaps:
        try:
            cmap = plt.get_cmap(name)
        except Exception:
            continue
        if getattr(cmap, 'colors', None) is not None:
            colors.extend(list(cmap.colors))
        else:
            colors.extend([cmap(i) for i in np.linspace(0.0, 1.0, 20)])
    if n <= len(colors):
        return colors[:n]
    cmap = plt.get_cmap('hsv', n)
    return [cmap(i) for i in range(n)]


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_figure(fig, out_base: Path, fmts: Sequence[str], dpi: int = 180):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in fmts:
        fmt = fmt.lower()
        out_path = out_base.with_suffix('.' + fmt)
        if fmt in {'png', 'jpg', 'jpeg'}:
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(out_path, bbox_inches='tight')
        saved.append(str(out_path))
    return saved


def _boxplot_compat(ax, values, labels, **kwargs):
    try:
        return ax.boxplot(values, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(values, labels=labels, **kwargs)


def compute_domain_stats(df: pd.DataFrame, group_col: str, embed_cols: Sequence[str]) -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError(f'Missing group column: {group_col}')

    work = df[[group_col] + list(embed_cols)].copy()
    work[group_col] = work[group_col].astype(str)
    grouped = work.groupby(group_col, sort=False)
    mean_df = grouped[list(embed_cols)].mean(numeric_only=True)
    var_df = grouped[list(embed_cols)].var(ddof=0, numeric_only=True)
    count_s = grouped.size().rename('n_spots')

    order = _sorted_group_labels(mean_df.index.tolist(), group_col)
    mean_df = mean_df.loc[order]
    var_df = var_df.loc[order]
    count_s = count_s.loc[order]

    data = {
        group_col: order,
        'n_spots': count_s.astype(int).to_numpy(),
    }
    for col in embed_cols:
        data[f'mean_{col}'] = mean_df[col].to_numpy(dtype=float)
    for col in embed_cols:
        vals = var_df[col].to_numpy(dtype=float)
        vals[np.isclose(vals, 0.0, atol=1e-12)] = 0.0
        data[f'var_{col}'] = vals
    return pd.DataFrame(data)


def save_domain_stats(stats_df: pd.DataFrame, out_path: Path):
    stats_df.to_csv(out_path, index=False, float_format='%.10f')
    print('Saved', out_path)


def plot_domain_overview(stats_df: pd.DataFrame, group_col: str, out_base: Path, fmts: Sequence[str]):
    mean_cols = [c for c in stats_df.columns if c.startswith('mean_h_')]
    var_cols = [c for c in stats_df.columns if c.startswith('var_h_')]
    labels = stats_df[group_col].astype(str).tolist()
    mean_mat = stats_df[mean_cols].to_numpy(dtype=float)
    var_mat = stats_df[var_cols].to_numpy(dtype=float)

    fig = plt.figure(figsize=(22, max(10, 0.42 * len(labels) + 8)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], width_ratios=[1, 1], hspace=0.28, wspace=0.18)

    ax_mean = fig.add_subplot(gs[0, 0])
    ax_var = fig.add_subplot(gs[0, 1])
    ax_box_mean = fig.add_subplot(gs[1, 0])
    ax_box_var = fig.add_subplot(gs[1, 1])

    max_abs_mean = float(np.nanmax(np.abs(mean_mat))) if mean_mat.size else 1.0
    im1 = ax_mean.imshow(mean_mat, aspect='auto', cmap='coolwarm', vmin=-max_abs_mean, vmax=max_abs_mean)
    ax_mean.set_title(f'{group_col}: mean heatmap')
    ax_mean.set_xlabel('Embedding dimension')
    ax_mean.set_ylabel(group_col)
    ax_mean.set_xticks(np.arange(mean_mat.shape[1]))
    ax_mean.set_xticklabels([str(i) for i in range(mean_mat.shape[1])], rotation=90, fontsize=7)
    ax_mean.set_yticks(np.arange(len(labels)))
    ax_mean.set_yticklabels(labels, fontsize=8)
    cbar1 = fig.colorbar(im1, ax=ax_mean, fraction=0.046, pad=0.02)
    cbar1.set_label('Mean')

    vmax_var = float(np.nanmax(var_mat)) if var_mat.size else 1.0
    im2 = ax_var.imshow(var_mat, aspect='auto', cmap='viridis', vmin=0.0, vmax=vmax_var)
    ax_var.set_title(f'{group_col}: variance heatmap')
    ax_var.set_xlabel('Embedding dimension')
    ax_var.set_ylabel(group_col)
    ax_var.set_xticks(np.arange(var_mat.shape[1]))
    ax_var.set_xticklabels([str(i) for i in range(var_mat.shape[1])], rotation=90, fontsize=7)
    ax_var.set_yticks(np.arange(len(labels)))
    ax_var.set_yticklabels(labels, fontsize=8)
    cbar2 = fig.colorbar(im2, ax=ax_var, fraction=0.046, pad=0.02)
    cbar2.set_label('Variance')

    mean_box = [mean_mat[i, :] for i in range(mean_mat.shape[0])]
    _boxplot_compat(
        ax_box_mean,
        mean_box,
        labels,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='#A7D3F2', edgecolor='#2A5B84'),
        medianprops=dict(color='#8B0000', linewidth=1.2),
        whiskerprops=dict(color='#2A5B84'),
        capprops=dict(color='#2A5B84'),
    )
    ax_box_mean.set_title(f'{group_col}: distribution of per-dimension means')
    ax_box_mean.set_xlabel(group_col)
    ax_box_mean.set_ylabel('Mean across h_0...h_N')
    ax_box_mean.tick_params(axis='x', rotation=60, labelsize=8)
    ax_box_mean.grid(axis='y', linestyle='--', alpha=0.35)

    var_box = [var_mat[i, :] for i in range(var_mat.shape[0])]
    _boxplot_compat(
        ax_box_var,
        var_box,
        labels,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='#B8E3B2', edgecolor='#2F6B2F'),
        medianprops=dict(color='#8B0000', linewidth=1.2),
        whiskerprops=dict(color='#2F6B2F'),
        capprops=dict(color='#2F6B2F'),
    )
    ax_box_var.set_title(f'{group_col}: distribution of per-dimension variances')
    ax_box_var.set_xlabel(group_col)
    ax_box_var.set_ylabel('Variance across h_0...h_N')
    ax_box_var.tick_params(axis='x', rotation=60, labelsize=8)
    ax_box_var.grid(axis='y', linestyle='--', alpha=0.35)

    fig.suptitle(f'Domain statistics overview by {group_col}', fontsize=15, y=0.995)
    paths = _save_figure(fig, out_base, fmts)
    plt.close(fig)
    for path in paths:
        print('Saved', path)


def save_selected_mean_table(stats_df: pd.DataFrame, group_col: str, selected_dims: Sequence[int], out_path: Path):
    cols = [group_col] + [f'mean_h_{d}' for d in selected_dims]
    stats_df[cols].to_csv(out_path, index=False, float_format='%.10f')
    print('Saved', out_path)


def plot_selected_mean_density(
    stats_by_group: Sequence[tuple[str, pd.DataFrame]],
    selected_dims: Sequence[int],
    out_base: Path,
    fmts: Sequence[str],
):
    fig, axes = plt.subplots(1, len(selected_dims), figsize=(6 * len(selected_dims), 4.8), constrained_layout=True)
    if len(selected_dims) == 1:
        axes = [axes]

    palette = {
        group_col: color for group_col, color in zip([name for name, _ in stats_by_group], ['#D55E00', '#0072B2', '#009E73', '#CC79A7'])
    }

    for ax, dim in zip(axes, selected_dims):
        col = f'mean_h_{dim}'
        all_vals = []
        for _, stats_df in stats_by_group:
            all_vals.extend(stats_df[col].astype(float).tolist())
        xmin, xmax = min(all_vals), max(all_vals)
        pad = max((xmax - xmin) * 0.12, 1e-3)
        xs = np.linspace(xmin - pad, xmax + pad, 400)

        for group_col, stats_df in stats_by_group:
            vals = stats_df[col].astype(float).to_numpy()
            if len(vals) >= 2 and np.std(vals) > 0:
                ys = gaussian_kde(vals)(xs)
            else:
                ys = np.zeros_like(xs)
            ax.plot(xs, ys, linewidth=2.2, color=palette[group_col], label=f'{group_col} (n={len(vals)})')
            ax.fill_between(xs, 0, ys, color=palette[group_col], alpha=0.16)

        ax.set_title(f'Smoothed density of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.legend(frameon=False, fontsize=9)

    paths = _save_figure(fig, out_base, fmts)
    plt.close(fig)
    for path in paths:
        print('Saved', path)


def plot_spot_level_kde(
    df: pd.DataFrame,
    group_col: str,
    selected_dims: Sequence[int],
    out_dir: Path,
    fmts: Sequence[str],
):
    if group_col not in df.columns:
        raise ValueError(f'Missing group column for KDE: {group_col}')

    labels = _sorted_group_labels(df[group_col].dropna().astype(str).unique().tolist(), group_col)
    palette = _categorical_palette(len(labels))
    color_map = {label: palette[i] for i, label in enumerate(labels)}

    work = df[[group_col] + [f'h_{d}' for d in selected_dims]].copy()
    work[group_col] = work[group_col].astype(str)

    for dim in selected_dims:
        col = f'h_{dim}'
        vals_all = pd.to_numeric(work[col], errors='coerce').dropna().to_numpy(dtype=float)
        xmin, xmax = float(vals_all.min()), float(vals_all.max())
        pad = max((xmax - xmin) * 0.08, 1e-3)
        xs = np.linspace(xmin - pad, xmax + pad, 500)

        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        for label in labels:
            vals = pd.to_numeric(work.loc[work[group_col] == label, col], errors='coerce').dropna().to_numpy(dtype=float)
            if vals.size < 2 or np.std(vals) == 0:
                continue
            ys = gaussian_kde(vals)(xs)
            ax.plot(xs, ys, linewidth=2.0, color=color_map[label], label=f'{label} (n={vals.size})')

        ax.set_title(f'Spot-level KDE of {col} by {group_col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.legend(frameon=False, fontsize=9, ncol=2)

        out_base = out_dir / f'{col}_spot_level_kde_by_{_sanitize_name(group_col)}'
        paths = _save_figure(fig, out_base, fmts)
        plt.close(fig)
        for path in paths:
            print('Saved', path)



def plot_joint_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    out_base: Path,
    fmts: Sequence[str],
    title: str,
    point_size: float = 2.0,
    alpha: float = 0.18,
):
    work = df[[x_col, y_col, group_col]].dropna().copy()
    work[group_col] = work[group_col].astype(str)
    labels = _sorted_group_labels(work[group_col].unique().tolist(), group_col)
    palette = _categorical_palette(len(labels))
    color_map = {label: palette[i] for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    for label in labels:
        sub = work[work[group_col] == label]
        ax.scatter(
            sub[x_col].to_numpy(),
            sub[y_col].to_numpy(),
            s=float(point_size),
            alpha=float(alpha),
            color=color_map[label],
            edgecolors='none',
            rasterized=True,
            label=f'{label} (n={len(sub)})',
        )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(linestyle='--', alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    paths = _save_figure(fig, out_base, fmts)
    plt.close(fig)
    for path in paths:
        print('Saved', path)


def sample_per_group(df: pd.DataFrame, group_col: str, max_per_group: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    work[group_col] = work[group_col].astype(str)
    labels = _sorted_group_labels(work[group_col].dropna().unique().tolist(), group_col)
    parts = []
    for label in labels:
        sub = work[work[group_col] == label]
        n = min(int(max_per_group), len(sub))
        parts.append(sub.sample(n=n, random_state=seed, replace=False))
    return pd.concat(parts, axis=0).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Analyze spot-level embedding domain statistics and plots.')
    parser.add_argument('--embeddings_csv', required=True, help='CSV produced by export_spot_embeddings.py')
    parser.add_argument('--out_dir', default=None, help='Output directory. Default: same directory as embeddings_csv')
    parser.add_argument('--embedding_prefix', default='h_', help='Embedding column prefix. Default: h_')
    parser.add_argument('--group_cols', nargs='+', default=list(DEFAULT_GROUP_COLS), help='Columns used as domains for summary stats')
    parser.add_argument('--selected_dims', type=int, nargs='+', default=list(DEFAULT_SELECTED_DIMS), help='Embedding dimensions for focused plots')
    parser.add_argument('--spot_kde_group_col', default='cancer_type', help='Group column used for spot-level KDE plots')
    parser.add_argument('--joint_scatter_group_col', default='cancer_type', help='Group column used for 2D scatter coloring')
    parser.add_argument('--joint_scatter_sample_caps', type=int, nargs='*', default=list(DEFAULT_JOINT_SCATTER_SAMPLE_CAPS), help='Balanced per-group sample caps for extra 2D scatter plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling and jitter')
    parser.add_argument('--formats', default='svg,png', help='Comma-separated output formats, e.g. svg,png')
    return parser


def run_analysis(args: argparse.Namespace):
    out_dir = _ensure_out_dir(Path(args.out_dir) if args.out_dir else Path(args.embeddings_csv).resolve().parent)
    df = pd.read_csv(args.embeddings_csv)
    embed_cols = _embedding_columns(df, args.embedding_prefix)
    selected_dims = [int(d) for d in args.selected_dims]
    if len(selected_dims) < 2:
        raise ValueError('selected_dims must contain at least 2 dimensions for joint scatter plotting')
    selected_cols = [f'{args.embedding_prefix}{d}' for d in selected_dims]
    for col in selected_cols:
        if col not in df.columns:
            raise ValueError(f'Selected embedding column not found: {col}')

    fmts = [x.strip() for x in str(args.formats).split(',') if x.strip()]

    stats_results = []
    for group_col in args.group_cols:
        stats_df = compute_domain_stats(df, group_col, embed_cols)
        stats_csv = out_dir / f'domain_stats_by_{_sanitize_name(group_col)}.csv'
        save_domain_stats(stats_df, stats_csv)
        plot_domain_overview(stats_df, group_col, out_dir / f'domain_stats_by_{_sanitize_name(group_col)}_overview', fmts)
        selected_csv = out_dir / f"selected_means_{_sanitize_name(group_col)}_{'_'.join(f'h{d}' for d in selected_dims)}.csv"
        save_selected_mean_table(stats_df, group_col, selected_dims, selected_csv)
        stats_results.append((group_col, stats_df))

    if len(stats_results) >= 2:
        density_name = 'mean_' + '_'.join(f'h{d}' for d in selected_dims) + '_density_compare'
        plot_selected_mean_density(stats_results, selected_dims, out_dir / density_name, fmts)

    plot_spot_level_kde(df, args.spot_kde_group_col, selected_dims, out_dir, fmts)

    x_col = f'{args.embedding_prefix}{selected_dims[0]}'
    y_col = f'{args.embedding_prefix}{selected_dims[1]}'
    plot_joint_scatter(
        df,
        x_col=x_col,
        y_col=y_col,
        group_col=args.joint_scatter_group_col,
        out_base=out_dir / f'{_sanitize_name(x_col)}_{_sanitize_name(y_col)}_spot_scatter_by_{_sanitize_name(args.joint_scatter_group_col)}',
        fmts=fmts,
        title=f'Spot-level scatter on ({x_col}, {y_col}) colored by {args.joint_scatter_group_col}',
        point_size=2.0,
        alpha=0.18,
    )

    if args.joint_scatter_sample_caps:
        base_cols = [x_col, y_col, args.joint_scatter_group_col]
        if 'spot_id' in df.columns:
            base_cols = ['spot_id'] + base_cols
        joint_df = df[base_cols].dropna().copy()
        for cap in args.joint_scatter_sample_caps:
            sampled = sample_per_group(joint_df, args.joint_scatter_group_col, max_per_group=int(cap), seed=int(args.seed))
            sampled_csv = out_dir / f'{_sanitize_name(x_col)}_{_sanitize_name(y_col)}_spot_scatter_by_{_sanitize_name(args.joint_scatter_group_col)}_sampled_{int(cap)}_per_group.csv'
            sampled.to_csv(sampled_csv, index=False)
            print('Saved', sampled_csv)
            plot_joint_scatter(
                sampled,
                x_col=x_col,
                y_col=y_col,
                group_col=args.joint_scatter_group_col,
                out_base=out_dir / f'{_sanitize_name(x_col)}_{_sanitize_name(y_col)}_spot_scatter_by_{_sanitize_name(args.joint_scatter_group_col)}_sampled_{int(cap)}_per_group',
                fmts=fmts,
                title=f'Sampled spot-level scatter on ({x_col}, {y_col}) by {args.joint_scatter_group_col}\nmax {int(cap)} spots per group',
                point_size=7.0 if int(cap) <= 1000 else 5.0,
                alpha=0.40 if int(cap) <= 1000 else 0.32,
            )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == '__main__':
    main()
