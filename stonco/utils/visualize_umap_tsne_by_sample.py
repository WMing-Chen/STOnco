import argparse
import os

import numpy as np
import pandas as pd


def _sample_rows(
    df: pd.DataFrame,
    max_points: int | None,
    seed: int,
    sample_frac: float | None = None,
    min_points_per_sample: int = 0,
    max_points_per_sample: int | None = None,
) -> pd.DataFrame:
    sampled = df

    if sample_frac is not None:
        sample_frac = float(sample_frac)
        min_points_per_sample = max(0, int(min_points_per_sample))
        max_points_per_sample = None if max_points_per_sample is None else max(1, int(max_points_per_sample))

        parts = []
        for _, group in df.groupby('sample_id', sort=True):
            n_total = len(group)
            n_keep = int(np.ceil(n_total * sample_frac))
            n_keep = max(min_points_per_sample, n_keep)
            if max_points_per_sample is not None:
                n_keep = min(max_points_per_sample, n_keep)
            n_keep = min(n_total, max(1, n_keep))
            if n_keep >= n_total:
                parts.append(group)
            else:
                parts.append(group.sample(n=n_keep, random_state=int(seed)))
        sampled = pd.concat(parts, axis=0).reset_index(drop=True)

    if max_points is None:
        return sampled
    max_points = int(max_points)
    if max_points <= 0 or len(sampled) <= max_points:
        return sampled.reset_index(drop=True)
    return sampled.sample(n=max_points, random_state=int(seed)).reset_index(drop=True)


def _get_embedding(df: pd.DataFrame, embed_source: str) -> np.ndarray:
    prefixes = [f'{embed_source}_']
    if embed_source == 'z_clf':
        prefixes.append('z64_')
    cols = []
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            break
    if len(cols) < 2:
        raise ValueError(f'Expected >=2 embedding columns for source {embed_source}, got {len(cols)}')
    cols = sorted(cols, key=lambda x: int(x.split('_')[-1]))
    return df[cols].to_numpy(dtype=float)


def _categorical_palette(n: int):
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
    umap_xy: np.ndarray,
    tsne_xy: np.ndarray,
    sample_ids: pd.Series,
    out_svg: str,
    point_size: float,
    alpha: float,
    title_fontsize: int,
    label_fontsize: int,
    legend_fontsize: int,
    highlight_mask: pd.Series | None = None,
    background_color: str = '#c8c8c8',
    background_alpha: float = 0.20,
    highlight_title_suffix: str | None = None,
) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sample_ids = sample_ids.astype(str)
    if highlight_mask is None:
        highlight_mask = pd.Series(np.ones(len(sample_ids), dtype=bool), index=sample_ids.index)
    else:
        highlight_mask = highlight_mask.astype(bool)
    bg_mask = ~highlight_mask.to_numpy()
    fg_mask = highlight_mask.to_numpy()

    highlight_sample_ids = sample_ids.loc[highlight_mask]
    uniq = sorted(highlight_sample_ids.unique().tolist())
    palette = _categorical_palette(len(uniq))
    label2color = {label: palette[i] for i, label in enumerate(uniq)}
    point_colors = highlight_sample_ids.map(label2color).to_list()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = f' ({highlight_title_suffix})' if highlight_title_suffix else ''
    panels = [
        (axes[0], umap_xy, f'UMAP by sample_id{title_suffix}', 'UMAP1', 'UMAP2'),
        (axes[1], tsne_xy, f't-SNE by sample_id{title_suffix}', 'tSNE1', 'tSNE2'),
    ]
    for ax, coords, title, xlabel, ylabel in panels:
        if np.any(bg_mask):
            ax.scatter(
                coords[bg_mask, 0],
                coords[bg_mask, 1],
                c=background_color,
                s=float(point_size),
                alpha=float(background_alpha),
                linewidths=0,
            )
        if np.any(fg_mask):
            ax.scatter(
                coords[fg_mask, 0],
                coords[fg_mask, 1],
                c=point_colors,
                s=float(point_size),
                alpha=float(alpha),
                linewidths=0,
            )
        ax.set_title(title, fontsize=int(title_fontsize))
        ax.set_xlabel(xlabel, fontsize=int(label_fontsize))
        ax.set_ylabel(ylabel, fontsize=int(label_fontsize))
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            label=label,
            markerfacecolor=label2color[label],
            markeredgecolor='none',
            markersize=7,
        )
        for label in uniq
    ]
    max_rows_per_col = 30
    ncol = max(1, int(np.ceil(len(handles) / max_rows_per_col)))
    legend_space = 0.14 if ncol == 1 else min(0.34, 0.12 + 0.07 * ncol)
    right = max(0.60, 1.0 - legend_space)
    if handles:
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


def main():
    parser = argparse.ArgumentParser(description='Visualize exported embeddings with UMAP + t-SNE colored by sample_id.')
    parser.add_argument('--embeddings_csv', required=True, help='CSV produced by export_spot_embeddings.py')
    parser.add_argument('--out_svg', default=None, help='Output SVG path (default: next to embeddings_csv)')
    parser.add_argument('--max_points', type=int, default=None, help='Optional subsample for speed (e.g. 50000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subsampling and DR reproducibility')
    parser.add_argument('--point_size', type=float, default=2.0, help='Scatter point size')
    parser.add_argument('--alpha', type=float, default=0.8, help='Scatter alpha')
    parser.add_argument('--title_fontsize', type=int, default=18, help='Title fontsize')
    parser.add_argument('--label_fontsize', type=int, default=14, help='Axis and sample label fontsize')
    parser.add_argument('--legend_fontsize', type=int, default=12, help='Legend fontsize')
    parser.add_argument('--sample_frac', type=float, default=None, help='Per-sample sampling fraction, e.g. 0.2')
    parser.add_argument('--min_points_per_sample', type=int, default=0, help='Minimum sampled spots for each sample_id')
    parser.add_argument('--max_points_per_sample', type=int, default=500, help='Maximum sampled spots for each sample_id')
    parser.add_argument('--highlight_cancer', default=None, help='Highlight only this cancer_type by sample_id; all other spots are shown in gray.')
    parser.add_argument('--background_color', default='#c8c8c8', help='Color for non-highlighted spots.')
    parser.add_argument('--background_alpha', type=float, default=0.20, help='Alpha for non-highlighted spots.')
    parser.add_argument(
        '--embed_source',
        choices=['h', 'z_clf', 'z64'],
        default='h',
        help='Which embedding columns to visualize: h_*, z_clf_*; z64 is kept as a legacy option.',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.embeddings_csv)
    if 'sample_id' not in df.columns:
        raise ValueError(f"Missing required column 'sample_id' in {args.embeddings_csv}")

    highlight_mask = None
    highlight_title_suffix = None
    if args.highlight_cancer is not None:
        if 'cancer_type' not in df.columns:
            raise ValueError("--highlight_cancer requires column 'cancer_type' in embeddings CSV")
        highlight_mask = df['cancer_type'].astype(str) == str(args.highlight_cancer)
        if not bool(highlight_mask.any()):
            raise ValueError(f"No rows found with cancer_type={args.highlight_cancer!r}")
        highlight_title_suffix = f'highlight {args.highlight_cancer}'

    df = _sample_rows(
        df,
        args.max_points,
        args.seed,
        sample_frac=args.sample_frac,
        min_points_per_sample=args.min_points_per_sample,
        max_points_per_sample=args.max_points_per_sample,
    )
    sample_ids = df['sample_id'].astype(str)
    if args.highlight_cancer is not None:
        highlight_mask = df['cancer_type'].astype(str) == str(args.highlight_cancer)
    Z = _get_embedding(df, args.embed_source)

    from sklearn.preprocessing import StandardScaler
    Zs = StandardScaler().fit_transform(Z)

    import umap
    umap_xy = umap.UMAP(n_components=2, random_state=int(args.seed)).fit_transform(Zs)

    from sklearn.manifold import TSNE
    tsne_xy = TSNE(
        n_components=2,
        random_state=int(args.seed),
        init='pca',
        learning_rate='auto',
    ).fit_transform(Zs)

    out_svg = args.out_svg
    if out_svg is None:
        out_dir = os.path.dirname(os.path.abspath(args.embeddings_csv)) or '.'
        out_svg = os.path.join(out_dir, f'umap_tsne_{args.embed_source}_by_sample.svg')

    _plot_two_panels(
        umap_xy,
        tsne_xy,
        sample_ids=sample_ids,
        out_svg=out_svg,
        point_size=args.point_size,
        alpha=args.alpha,
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        legend_fontsize=args.legend_fontsize,
        highlight_mask=highlight_mask,
        background_color=args.background_color,
        background_alpha=args.background_alpha,
        highlight_title_suffix=highlight_title_suffix,
    )
    print('Saved', out_svg)


if __name__ == '__main__':
    main()
