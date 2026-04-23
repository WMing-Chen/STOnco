import argparse
import os
import re

import numpy as np
import pandas as pd

from stonco.utils.embedding_analysis import (
    attach_reduction_columns,
    compute_reductions,
    get_embedding_matrix,
    sample_rows,
)


def _sanitize_name(text: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text).strip())
    return safe.strip('_') or 'column'


def _resolve_metadata_column(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    key = str(col).strip().lower().replace('-', '_')
    aliases = {
        'cancer': 'cancer_type',
        'cancer_type': 'cancer_type',
        'sample': 'sample_id',
        'sampleid': 'sample_id',
        'sample_id': 'sample_id',
        'batch': 'batch_id',
        'batchid': 'batch_id',
        'batch_id': 'batch_id',
    }
    return aliases.get(key, col)


def _expand_cli_values(values) -> list[str]:
    expanded = []
    for value in values or []:
        for part in str(value).split(','):
            part = part.strip()
            if part:
                expanded.append(part)
    return expanded


def _normalize_category_labels(color_col: str, series: pd.Series) -> pd.Series:
    s = series.copy()
    try:
        s = s.fillna('NA')
    except Exception:
        pass
    s = s.astype(str)
    s = s.replace({'nan': 'NA', 'None': 'NA'})
    s = s.replace(r'^\s*$', 'NA', regex=True)

    if color_col == 'tumor_label':
        num = pd.to_numeric(series, errors='coerce')
        labels = pd.Series(['NA'] * len(series), index=series.index, dtype=object)
        labels[num == 0] = 'Non-malignant spot'
        labels[num == 1] = 'Malignant spot'
        labels[pd.isna(num)] = 'NA'
        return labels

    return s


def _categorical_palette(n: int):
    import matplotlib
    import matplotlib.pyplot as plt

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


def _plot_two_panels(
    coords_left,
    coords_right,
    title_left,
    title_right,
    color_col,
    color_series,
    out_svg,
    point_size=2.0,
    alpha=0.8,
    title_fontsize=18,
    label_fontsize=14,
    legend_fontsize=12,
    highlight_mask=None,
    highlight_label=None,
    highlight_marker='^',
    highlight_point_size=None,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = _normalize_category_labels(color_col, color_series)
    uniq = sorted(labels.unique().tolist())
    if color_col == 'tumor_label':
        desired = ['Non-malignant spot', 'Malignant spot', 'NA']
        uniq = [u for u in desired if u in set(uniq)] + [u for u in uniq if u not in set(desired)]

    palette = _categorical_palette(len(uniq))
    label2color = {u: palette[i] for i, u in enumerate(uniq)}
    if color_col == 'tumor_label':
        if 'Non-malignant spot' in label2color:
            label2color['Non-malignant spot'] = '#71CCEA'
        if 'Malignant spot' in label2color:
            label2color['Malignant spot'] = '#B20A0A'
    point_colors = np.asarray(labels.map(label2color).to_list(), dtype=object)
    if highlight_mask is None:
        highlight_mask = np.zeros(len(labels), dtype=bool)
    else:
        highlight_mask = np.asarray(highlight_mask, dtype=bool)
        if highlight_mask.shape[0] != len(labels):
            raise ValueError(f'highlight_mask length mismatch: {highlight_mask.shape[0]} vs {len(labels)}')
    normal_mask = ~highlight_mask
    highlight_point_size = float(highlight_point_size if highlight_point_size is not None else max(float(point_size) * 3.0, 10.0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in zip(axes, [coords_left, coords_right], [title_left, title_right]):
        if np.any(normal_mask):
            ax.scatter(
                coords[normal_mask, 0],
                coords[normal_mask, 1],
                c=point_colors[normal_mask].tolist(),
                s=float(point_size),
                alpha=float(alpha),
                linewidths=0,
                marker='o',
            )
        if np.any(highlight_mask):
            highlight_scatter = ax.scatter(
                coords[highlight_mask, 0],
                coords[highlight_mask, 1],
                facecolors=point_colors[highlight_mask].tolist(),
                s=highlight_point_size,
                alpha=float(alpha),
                linewidths=0,
                edgecolors='none',
                marker=highlight_marker,
            )
            highlight_scatter.set_edgecolor('none')
            highlight_scatter.set_linewidth(0)
        ax.set_title(title, fontsize=int(title_fontsize))
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_xlabel('UMAP1', fontsize=int(label_fontsize))
    axes[0].set_ylabel('UMAP2', fontsize=int(label_fontsize))
    axes[1].set_xlabel('tSNE1', fontsize=int(label_fontsize))
    axes[1].set_ylabel('tSNE2', fontsize=int(label_fontsize))

    handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            label=u,
            markerfacecolor=label2color[u],
            markeredgecolor='none',
            markersize=7,
        )
        for u in uniq
    ]
    if np.any(highlight_mask):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=highlight_marker,
                color='none',
                label=highlight_label or 'Highlighted',
                markerfacecolor='#6e6e6e',
                markeredgecolor='none',
                markersize=8,
                linewidth=0,
            )
        )
    max_rows_per_col = 30
    ncol = max(1, int(np.ceil(len(handles) / max_rows_per_col)))

    legend_space = 0.14 if ncol == 1 else min(0.34, 0.12 + 0.07 * ncol)
    right = max(0.60, 1.0 - legend_space)
    fig.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(right + 0.01, 0.5),
        frameon=False,
        fontsize=int(legend_fontsize),
        ncol=ncol,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0.0, 0.0, right, 1.0])
    os.makedirs(os.path.dirname(out_svg) or '.', exist_ok=True)
    fig.savefig(out_svg, format='svg', dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return out_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize exported h/classifier-latent embeddings with UMAP + t-SNE.')
    parser.add_argument('--embeddings_csv', required=True, help='CSV produced by export_spot_embeddings.py')
    parser.add_argument('--out_dir', default=None, help='Output directory for SVGs (default: same as embeddings_csv)')
    parser.add_argument(
        '--max_points',
        type=int,
        default=None,
        help='Optional subsample for speed. If unset and sample_id exists, defaults to 500 points per sample.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subsampling and DR reproducibility')
    parser.add_argument('--point_size', type=float, default=2.0, help='Scatter point size')
    parser.add_argument('--alpha', type=float, default=0.8, help='Scatter alpha')
    parser.add_argument(
        '--embed_source',
        choices=['h', 'z_clf', 'z64'],
        default='h',
        help='Which embedding columns to visualize: h_*, z_clf_*; z64 is kept as a legacy option.',
    )
    parser.add_argument(
        '--color_cols',
        nargs='+',
        default=None,
        help='Optional metadata columns used for coloring. Default: tumor_label batch_id cancer_type',
    )
    parser.add_argument('--out_coords_csv', default=None, help='Optional CSV path to save sampled rows with UMAP/t-SNE coordinates.')
    parser.add_argument('--umap_n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--umap_min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--tsne_perplexity', type=float, default=None, help='Optional t-SNE perplexity override')
    parser.add_argument(
        '--highlight_col',
        default=None,
        help='Optional metadata column used to draw selected spots as triangles, e.g. cancer_type, sample_id, or batch_id.',
    )
    parser.add_argument(
        '--highlight_values',
        nargs='+',
        default=None,
        help='Values in --highlight_col to draw as triangles. Other spots keep the default circle marker.',
    )
    parser.add_argument('--highlight_marker', default='^', help='Matplotlib marker for highlighted spots. Default: triangle (^).')
    parser.add_argument('--highlight_point_size', type=float, default=None, help='Optional marker size for highlighted spots.')
    return parser


def run_visualization(args: argparse.Namespace) -> dict[str, object]:
    df = pd.read_csv(args.embeddings_csv)
    if args.max_points is None and 'sample_id' in df.columns:
        args.max_points = 500 * int(df['sample_id'].astype(str).nunique())
    df = sample_rows(df, args.max_points, args.seed)
    Z = get_embedding_matrix(df, args.embed_source)
    reductions = compute_reductions(
        Z,
        seed=args.seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        tsne_perplexity=args.tsne_perplexity,
        run_umap=True,
        run_tsne=True,
    )
    umap_xy = reductions['umap']
    tsne_xy = reductions['tsne']

    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.embeddings_csv)) or '.')
    os.makedirs(out_dir, exist_ok=True)

    highlight_mask = None
    highlight_label = None
    if args.highlight_col or args.highlight_values:
        if not args.highlight_col or not args.highlight_values:
            raise ValueError('--highlight_col and --highlight_values must be provided together.')
        highlight_col = _resolve_metadata_column(df, args.highlight_col)
        if highlight_col not in df.columns:
            print(f'[Warn] Missing highlight column {args.highlight_col} in embeddings CSV; no triangle highlight will be used.')
        else:
            highlight_values = _expand_cli_values(args.highlight_values)
            values = {str(v) for v in highlight_values}
            highlight_mask = df[highlight_col].astype(str).isin(values).to_numpy()
            value_text = ','.join(str(v) for v in highlight_values)
            highlight_label = f'{highlight_col}={value_text}'
            print(f'[Info] Highlight {int(highlight_mask.sum())}/{len(df)} spots as triangles: {highlight_label}')

    color_cols = args.color_cols or ['tumor_label', 'batch_id', 'cancer_type']
    plots = [(col, f'umap_tsne_{args.embed_source}_by_{_sanitize_name(col)}.svg') for col in color_cols]
    for col, fname in plots:
        if col not in df.columns:
            print(f'[Warn] Missing column {col} in embeddings CSV, skip.')
            continue
        out_svg = os.path.join(out_dir, fname)
        _plot_two_panels(
            umap_xy,
            tsne_xy,
            title_left=f'UMAP colored by {col}',
            title_right=f't-SNE colored by {col}',
            color_col=col,
            color_series=df[col],
            out_svg=out_svg,
            point_size=args.point_size,
            alpha=args.alpha,
            highlight_mask=highlight_mask,
            highlight_label=highlight_label,
            highlight_marker=args.highlight_marker,
            highlight_point_size=args.highlight_point_size,
        )
        print('Saved', out_svg)

    out_coords_csv = args.out_coords_csv
    if out_coords_csv:
        out_coords_csv = os.path.abspath(out_coords_csv)
        os.makedirs(os.path.dirname(out_coords_csv) or '.', exist_ok=True)
        coords_df = attach_reduction_columns(df, reductions)
        coords_df.to_csv(out_coords_csv, index=False, float_format='%.6f')
        print('Saved', out_coords_csv)

    return {
        'out_dir': out_dir,
        'out_coords_csv': out_coords_csv,
        'n_points': int(len(df)),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_visualization(args)


if __name__ == '__main__':
    main()
