from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from stonco.utils.embedding_analysis import (
    ROLE_TO_METRIC,
    SPACE_CHOICES,
    build_knn_indices,
    compute_lisi_scores,
    parse_group_roles,
    prepare_space_data,
    sample_rows,
    summarize_scores,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate embedding mixing with iLISI/cLISI on embedding or DR spaces.')
    parser.add_argument('--embeddings_csv', required=True, help='Main CSV containing metadata and embedding columns.')
    parser.add_argument(
        '--coords_csv',
        default=None,
        help='Optional CSV containing UMAP/t-SNE coordinates. If omitted, requested coord columns must exist in embeddings_csv.',
    )
    parser.add_argument('--out_csv', required=True, help='Output summary CSV path.')
    parser.add_argument('--out_spot_csv', default=None, help='Optional output CSV path for per-spot LISI scores.')
    parser.add_argument(
        '--embed_source',
        choices=['h', 'z_clf', 'z64'],
        default='h',
        help='Embedding source used when evaluating embedding space.',
    )
    parser.add_argument(
        '--spaces',
        nargs='+',
        default=['embedding'],
        choices=list(SPACE_CHOICES),
        help='Spaces to evaluate. embedding is the primary metric space; umap/tsne are auxiliary.',
    )
    parser.add_argument('--group_cols', nargs='+', required=True, help='Metadata columns to evaluate.')
    parser.add_argument(
        '--group_roles',
        nargs='*',
        default=None,
        help='Explicit column-role mappings, e.g. sample_id:integration tumor_label:conservation',
    )
    parser.add_argument('--k_values', nargs='+', type=int, default=[15, 30, 50], help='Neighborhood sizes used for LISI.')
    parser.add_argument('--max_points', type=int, default=None, help='Optional subsample cap applied consistently per input CSV.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used when sampling rows.')
    return parser


def _load_space_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    base_df = pd.read_csv(args.embeddings_csv)
    base_df = sample_rows(base_df, args.max_points, args.seed)
    frames = {'embedding': base_df}

    if any(space in {'umap', 'tsne'} for space in args.spaces):
        coords_path = args.coords_csv or args.embeddings_csv
        coords_df = pd.read_csv(coords_path)
        coords_df = sample_rows(coords_df, args.max_points, args.seed)
        frames['umap'] = coords_df
        frames['tsne'] = coords_df
    return frames


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    role_map = parse_group_roles(args.group_cols, args.group_roles)
    frames = _load_space_frames(args)
    k_values = sorted({int(k) for k in args.k_values if int(k) > 0})
    if not k_values:
        raise ValueError('At least one positive k is required')

    summary_rows: list[dict[str, object]] = []
    spot_rows: list[dict[str, object]] = []

    for space in args.spaces:
        frame = frames[space]
        space_data = prepare_space_data(frame, space=space, embed_source=args.embed_source)
        max_k = min(max(k_values), max(1, len(space_data.df) - 1))
        if max_k <= 0:
            raise ValueError(f'Not enough rows to evaluate space {space!r}')
        knn_indices = build_knn_indices(space_data.matrix, max_k=max_k)

        for group_col in args.group_cols:
            if group_col not in space_data.df.columns:
                print(f'[Warn] Missing column {group_col} in {space} data, skip.')
                continue
            role = role_map[group_col]
            metric_name = ROLE_TO_METRIC[role]
            series = space_data.df[group_col]
            for k in k_values:
                k_used = min(int(k), knn_indices.shape[1])
                scores, effective_neighbors, n_groups = compute_lisi_scores(series, knn_indices[:, :k_used])
                if n_groups <= 1:
                    print(f'[Warn] Column {group_col} has <=1 valid groups in {space} space, skip k={k_used}.')
                    continue

                summary_rows.append(
                    summarize_scores(
                        scores,
                        group_col=group_col,
                        group_role=role,
                        metric_name=metric_name,
                        space=space,
                        embed_source=args.embed_source,
                        embed_dim=space_data.dim,
                        k=k_used,
                        n_spots_total=len(space_data.df),
                        n_groups=n_groups,
                    )
                )

                if args.out_spot_csv:
                    spot_ids = (
                        space_data.df['spot_id'].astype(str).to_numpy()
                        if 'spot_id' in space_data.df.columns
                        else np.array([str(i) for i in range(len(space_data.df))], dtype=object)
                    )
                    sample_ids = (
                        space_data.df['sample_id'].astype(str).to_numpy()
                        if 'sample_id' in space_data.df.columns
                        else np.array(['NA'] * len(space_data.df), dtype=object)
                    )
                    for idx in range(len(space_data.df)):
                        spot_rows.append(
                            {
                                'spot_id': spot_ids[idx],
                                'sample_id': sample_ids[idx],
                                'group_col': group_col,
                                'group_role': role,
                                'metric_name': metric_name,
                                'space': space,
                                'k': int(k_used),
                                'effective_neighbors': int(effective_neighbors[idx]),
                                'lisi': float(scores[idx]) if np.isfinite(scores[idx]) else np.nan,
                            }
                        )

    if not summary_rows:
        raise ValueError('No valid LISI results were generated. Check group columns and requested spaces.')

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['space', 'group_col', 'k']).reset_index(drop=True)
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    summary_df.to_csv(out_csv, index=False, float_format='%.6f')
    print('Saved', out_csv)

    out_spot_csv = None
    if args.out_spot_csv:
        out_spot_csv = os.path.abspath(args.out_spot_csv)
        os.makedirs(os.path.dirname(out_spot_csv) or '.', exist_ok=True)
        pd.DataFrame(spot_rows).to_csv(out_spot_csv, index=False, float_format='%.6f')
        print('Saved', out_spot_csv)

    return {
        'summary_df': summary_df,
        'out_csv': out_csv,
        'out_spot_csv': out_spot_csv,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()
