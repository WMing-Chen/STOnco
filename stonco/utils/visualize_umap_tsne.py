import argparse
import os

import numpy as np
import pandas as pd


def _sample_rows(df: pd.DataFrame, max_points: int | None, seed: int) -> pd.DataFrame:
    if max_points is None:
        return df
    max_points = int(max_points)
    if max_points <= 0 or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=int(seed)).reset_index(drop=True)


def _get_z64(df: pd.DataFrame) -> np.ndarray:
    z_cols = [c for c in df.columns if c.startswith('z64_')]
    if len(z_cols) != 64:
        raise ValueError(f'Expected 64 columns starting with z64_, got {len(z_cols)}')
    z_cols = sorted(z_cols, key=lambda x: int(x.split('_')[-1]))
    return df[z_cols].to_numpy(dtype=float)


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
    point_colors = labels.map(label2color).to_list()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in zip(axes, [coords_left, coords_right], [title_left, title_right]):
        ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=float(point_size), alpha=float(alpha), linewidths=0)
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


def main():
    parser = argparse.ArgumentParser(description='Visualize exported z64 embeddings with UMAP + t-SNE.')
    parser.add_argument('--embeddings_csv', required=True, help='CSV produced by export_spot_embeddings.py')
    parser.add_argument('--out_dir', default=None, help='Output directory for SVGs (default: same as embeddings_csv)')
    parser.add_argument('--max_points', type=int, default=None, help='Optional subsample for speed (e.g. 50000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subsampling and DR reproducibility')
    parser.add_argument('--point_size', type=float, default=2.0, help='Scatter point size')
    parser.add_argument('--alpha', type=float, default=0.8, help='Scatter alpha')
    args = parser.parse_args()

    df = pd.read_csv(args.embeddings_csv)
    df = _sample_rows(df, args.max_points, args.seed)
    Z = _get_z64(df)

    from sklearn.preprocessing import StandardScaler
    Zs = StandardScaler().fit_transform(Z)

    # UMAP
    import umap
    umap_xy = umap.UMAP(n_components=2, random_state=int(args.seed)).fit_transform(Zs)

    # t-SNE
    from sklearn.manifold import TSNE
    tsne_xy = TSNE(
        n_components=2,
        random_state=int(args.seed),
        init='pca',
        learning_rate='auto',
    ).fit_transform(Zs)

    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.embeddings_csv)) or '.')
    os.makedirs(out_dir, exist_ok=True)

    plots = [
        ('tumor_label', 'umap_tsne_by_tumor.svg'),
        ('batch_id', 'umap_tsne_by_batch.svg'),
        ('cancer_type', 'umap_tsne_by_cancer.svg'),
    ]
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
        )
        print('Saved', out_svg)


if __name__ == '__main__':
    main()
