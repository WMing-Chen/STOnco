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


def _encode_categories(series: pd.Series):
    s = series.astype(str).fillna('NA')
    uniq = sorted(s.unique().tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    codes = s.map(mapping).to_numpy(dtype=int)
    return codes, uniq


def _plot_two_panels(coords_left, coords_right, title_left, title_right, color_series, out_svg, point_size=2.0, alpha=0.8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    codes, uniq = _encode_categories(color_series)
    cmap = 'tab20'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in zip(axes, [coords_left, coords_right], [title_left, title_right]):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=codes, cmap=cmap, s=float(point_size), alpha=float(alpha), linewidths=0)
        ax.set_title(title)
        ax.set_xlabel('dim1')
        ax.set_ylabel('dim2')
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if len(uniq) <= 20:
            cb.set_ticks(list(range(len(uniq))))
            cb.set_ticklabels(uniq)
        else:
            cb.set_label('category_code')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_svg) or '.', exist_ok=True)
    fig.savefig(out_svg, format='svg', dpi=150)
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
            color_series=df[col],
            out_svg=out_svg,
            point_size=args.point_size,
            alpha=args.alpha,
        )
        print('Saved', out_svg)


if __name__ == '__main__':
    main()

