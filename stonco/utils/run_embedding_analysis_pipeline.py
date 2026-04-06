from __future__ import annotations

import argparse
import os
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run export + UMAP/t-SNE visualization + LISI evaluation in one pipeline.')
    parser.add_argument('--artifacts_dir', required=True, help='Training artifacts dir used by export_spot_embeddings.py')
    parser.add_argument('--out_dir', required=True, help='Output directory for exported embeddings, plots, coords, and metrics')
    parser.add_argument('--train_npz', default=None, help='Multi-slide NPZ with Xs/xys/ys/slide_ids/gene_names.')
    parser.add_argument('--npz', action='append', default=None, help='Single-slide NPZ path. Can be repeated.')
    parser.add_argument('--npz_glob', action='append', default=None, help='Glob for single-slide NPZ files. Can be repeated.')
    parser.add_argument('--subset', choices=['all', 'train', 'val'], default='all', help='Subset used when --train_npz is provided')
    parser.add_argument('--device', default=None, help='cpu/cuda; default auto')
    parser.add_argument('--num_threads', type=int, default=None, help='Torch CPU threads for export')
    parser.add_argument(
        '--embed_source',
        choices=['h', 'z_clf', 'z64'],
        default='h',
        help='Embedding source used in export, visualization, and evaluation.',
    )
    parser.add_argument('--group_cols', nargs='+', required=True, help='Metadata columns to evaluate with LISI.')
    parser.add_argument(
        '--group_roles',
        nargs='*',
        default=None,
        help='Explicit column-role mappings, e.g. sample_id:integration tumor_label:conservation',
    )
    parser.add_argument(
        '--metric_spaces',
        nargs='+',
        default=['embedding', 'umap'],
        choices=['embedding', 'umap', 'tsne'],
        help='Spaces passed to evaluate_embedding_mixing.',
    )
    parser.add_argument('--k_values', nargs='+', type=int, default=[15, 30, 50], help='Neighborhood sizes used for LISI.')
    parser.add_argument('--max_points', type=int, default=None, help='Optional shared row cap used by visualization/evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for DR and sampling.')
    parser.add_argument('--point_size', type=float, default=2.0, help='Scatter point size for plots.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Scatter alpha for plots.')
    parser.add_argument(
        '--color_cols',
        nargs='+',
        default=['sample_id', 'batch_id', 'cancer_type', 'tumor_label'],
        help='Metadata columns used for SVG coloring.',
    )
    parser.add_argument('--umap_n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter.')
    parser.add_argument('--umap_min_dist', type=float, default=0.1, help='UMAP min_dist parameter.')
    parser.add_argument('--tsne_perplexity', type=float, default=None, help='Optional t-SNE perplexity override.')
    parser.add_argument('--save_spot_metrics', action='store_true', help='Also save per-spot LISI values.')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base_name = f'{args.embed_source}'
    embeddings_csv = os.path.join(out_dir, f'spot_embeddings_{base_name}.csv')
    coords_csv = os.path.join(out_dir, f'spot_embeddings_{base_name}_coords.csv')
    metrics_csv = os.path.join(out_dir, f'mixing_metrics_{base_name}.csv')
    spot_metrics_csv = os.path.join(out_dir, f'mixing_metrics_{base_name}_spot.csv')

    export_cmd = [
        sys.executable,
        '-m',
        'stonco.utils.export_spot_embeddings',
        '--artifacts_dir',
        args.artifacts_dir,
        '--out_csv',
        embeddings_csv,
        '--subset',
        args.subset,
        '--embed_source',
        args.embed_source,
    ]
    if args.train_npz:
        export_cmd.extend(['--train_npz', args.train_npz])
    for path in args.npz or []:
        export_cmd.extend(['--npz', path])
    for pattern in args.npz_glob or []:
        export_cmd.extend(['--npz_glob', pattern])
    if args.device:
        export_cmd.extend(['--device', args.device])
    if args.num_threads is not None:
        export_cmd.extend(['--num_threads', str(args.num_threads)])
    print('Running:', ' '.join(export_cmd))
    subprocess.run(export_cmd, check=True)

    visualize_cmd = [
        sys.executable,
        '-m',
        'stonco.utils.visualize_umap_tsne',
        '--embeddings_csv',
        embeddings_csv,
        '--out_dir',
        out_dir,
        '--out_coords_csv',
        coords_csv,
        '--embed_source',
        args.embed_source,
        '--seed',
        str(args.seed),
        '--point_size',
        str(args.point_size),
        '--alpha',
        str(args.alpha),
        '--umap_n_neighbors',
        str(args.umap_n_neighbors),
        '--umap_min_dist',
        str(args.umap_min_dist),
        '--color_cols',
        *args.color_cols,
    ]
    if args.max_points is not None:
        visualize_cmd.extend(['--max_points', str(args.max_points)])
    if args.tsne_perplexity is not None:
        visualize_cmd.extend(['--tsne_perplexity', str(args.tsne_perplexity)])
    print('Running:', ' '.join(visualize_cmd))
    subprocess.run(visualize_cmd, check=True)

    metrics_input_csv = coords_csv if os.path.exists(coords_csv) else embeddings_csv
    evaluate_cmd = [
        sys.executable,
        '-m',
        'stonco.utils.evaluate_embedding_mixing',
        '--embeddings_csv',
        metrics_input_csv,
        '--out_csv',
        metrics_csv,
        '--embed_source',
        args.embed_source,
        '--spaces',
        *args.metric_spaces,
        '--group_cols',
        *args.group_cols,
        '--k_values',
        *[str(k) for k in args.k_values],
        '--seed',
        str(args.seed),
    ]
    if os.path.exists(coords_csv):
        evaluate_cmd.extend(['--coords_csv', coords_csv])
    if args.group_roles:
        evaluate_cmd.extend(['--group_roles', *args.group_roles])
    if args.save_spot_metrics:
        evaluate_cmd.extend(['--out_spot_csv', spot_metrics_csv])
    print('Running:', ' '.join(evaluate_cmd))
    subprocess.run(evaluate_cmd, check=True)


if __name__ == '__main__':
    main()
