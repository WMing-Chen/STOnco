from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROLE_TO_METRIC = {
    'integration': 'iLISI',
    'conservation': 'cLISI',
}

SPACE_CHOICES = ('embedding', 'umap', 'tsne')
COMMON_INTEGRATION_COLS = {'sample_id', 'batch_id', 'slide_id'}
COMMON_CONSERVATION_COLS = {'tumor_label', 'cell_type', 'region'}


@dataclass(frozen=True)
class SpaceData:
    df: pd.DataFrame
    matrix: np.ndarray
    space: str
    dim: int


def sample_rows(df: pd.DataFrame, max_points: int | None, seed: int) -> pd.DataFrame:
    if max_points is None:
        return df.reset_index(drop=True)
    max_points = int(max_points)
    if max_points <= 0 or len(df) <= max_points:
        return df.reset_index(drop=True)
    if 'sample_id' not in df.columns:
        return df.sample(n=max_points, random_state=int(seed)).reset_index(drop=True)

    sample_frac = float(max_points) / float(len(df))
    parts = []
    for _, group in df.groupby('sample_id', sort=True):
        n_keep = int(np.ceil(len(group) * sample_frac))
        n_keep = min(len(group), max(1, n_keep))
        if n_keep >= len(group):
            parts.append(group)
        else:
            parts.append(group.sample(n=n_keep, random_state=int(seed)))
    return pd.concat(parts, axis=0).reset_index(drop=True)


def get_embedding_columns(df: pd.DataFrame, embed_source: str) -> list[str]:
    prefixes = [f'{embed_source}_']
    if embed_source == 'z_clf':
        prefixes.append('z64_')
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            return sorted(cols, key=lambda x: int(x.split('_')[-1]))
    return []


def get_embedding_matrix(df: pd.DataFrame, embed_source: str) -> np.ndarray:
    cols = get_embedding_columns(df, embed_source)
    if len(cols) < 2:
        raise ValueError(f'Expected >=2 embedding columns for source {embed_source}, got {len(cols)}')
    return df[cols].to_numpy(dtype=float)


def get_space_matrix(df: pd.DataFrame, space: str, embed_source: str) -> np.ndarray:
    space = str(space)
    if space == 'embedding':
        return get_embedding_matrix(df, embed_source)
    if space == 'umap':
        cols = ['umap_1', 'umap_2']
    elif space == 'tsne':
        cols = ['tsne_1', 'tsne_2']
    else:
        raise ValueError(f'Unsupported space: {space}')
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns for {space} space: {missing}')
    return df[cols].to_numpy(dtype=float)


def prepare_space_data(df: pd.DataFrame, space: str, embed_source: str) -> SpaceData:
    X = get_space_matrix(df, space=space, embed_source=embed_source)
    if space == 'embedding':
        X = StandardScaler().fit_transform(X)
    return SpaceData(df=df.reset_index(drop=True), matrix=X, space=space, dim=int(X.shape[1]))


def choose_tsne_perplexity(n_samples: int, requested: float | None = None) -> float:
    if n_samples < 3:
        raise ValueError('t-SNE requires at least 3 samples')
    if requested is None:
        requested = 30.0
    upper = max(2.0, float(n_samples - 1) / 3.0)
    return float(min(float(requested), upper))


def compute_reductions(
    embedding_matrix: np.ndarray,
    seed: int,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float | None = None,
    run_umap: bool = True,
    run_tsne: bool = True,
) -> dict[str, np.ndarray]:
    Zs = StandardScaler().fit_transform(np.asarray(embedding_matrix, dtype=float))
    out: dict[str, np.ndarray] = {}

    if run_umap:
        import umap

        out['umap'] = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            random_state=int(seed),
        ).fit_transform(Zs)

    if run_tsne:
        from sklearn.manifold import TSNE

        perplexity = choose_tsne_perplexity(Zs.shape[0], requested=tsne_perplexity)
        out['tsne'] = TSNE(
            n_components=2,
            random_state=int(seed),
            init='pca',
            learning_rate='auto',
            perplexity=float(perplexity),
        ).fit_transform(Zs)

    return out


def attach_reduction_columns(
    df: pd.DataFrame,
    reductions: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    if 'umap' in reductions:
        out['umap_1'] = reductions['umap'][:, 0]
        out['umap_2'] = reductions['umap'][:, 1]
    if 'tsne' in reductions:
        out['tsne_1'] = reductions['tsne'][:, 0]
        out['tsne_2'] = reductions['tsne'][:, 1]
    return out


def build_knn_indices(matrix: np.ndarray, max_k: int) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'Expected 2D matrix, got shape {X.shape}')
    n_samples = int(X.shape[0])
    if n_samples < 2:
        raise ValueError('At least 2 rows are required to build kNN neighborhoods')
    max_k = int(max_k)
    if max_k <= 0:
        raise ValueError(f'max_k must be > 0, got {max_k}')
    n_neighbors = min(n_samples, max_k + 1)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X)
    indices = knn.kneighbors(X, return_distance=False)
    if indices.shape[1] > 1:
        return indices[:, 1:]
    return np.empty((n_samples, 0), dtype=int)


def normalize_group_labels(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.astype(object)
        s = s.where(~pd.isna(s), np.nan)
        s = s.map(lambda v: np.nan if v is None else str(v).strip())
        s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NA': np.nan})
    return s


def infer_group_role(group_col: str) -> str:
    col = str(group_col)
    if col in COMMON_INTEGRATION_COLS:
        return 'integration'
    if col in COMMON_CONSERVATION_COLS:
        return 'conservation'
    raise ValueError(
        f'Cannot infer role for group column {group_col!r}. '
        'Please provide --group_roles with explicit mappings such as sample_id:integration.'
    )


def parse_group_roles(group_cols: Iterable[str], group_roles: Iterable[str] | None) -> dict[str, str]:
    role_map: dict[str, str] = {}
    if group_roles:
        for item in group_roles:
            text = str(item).strip()
            if ':' not in text:
                raise ValueError(f'Invalid group role mapping {item!r}; expected col:role')
            col, role = text.split(':', 1)
            col = col.strip()
            role = role.strip()
            if role not in ROLE_TO_METRIC:
                raise ValueError(f'Unsupported role {role!r} for column {col!r}')
            role_map[col] = role

    for col in group_cols:
        if col not in role_map:
            role_map[col] = infer_group_role(col)
    return role_map


def _encode_labels(series: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    normalized = normalize_group_labels(series)
    valid_mask = ~pd.isna(normalized)
    encoded = np.full(len(normalized), -1, dtype=int)
    if valid_mask.any():
        codes, uniques = pd.factorize(normalized.loc[valid_mask], sort=True)
        encoded[valid_mask.to_numpy()] = codes.astype(int)
        n_groups = int(len(uniques))
    else:
        n_groups = 0
    return encoded, valid_mask.to_numpy(), n_groups


def compute_lisi_scores(
    series: pd.Series,
    knn_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    encoded, center_valid_mask, n_groups = _encode_labels(series)
    scores = np.full(len(series), np.nan, dtype=float)
    effective_neighbors = np.zeros(len(series), dtype=int)
    if n_groups <= 1 or knn_indices.shape[1] == 0:
        return scores, effective_neighbors, n_groups

    for i in range(len(series)):
        if not center_valid_mask[i]:
            continue
        labels = encoded[knn_indices[i]]
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        counts = np.bincount(labels)
        probs = counts[counts > 0].astype(float)
        probs /= float(labels.size)
        scores[i] = 1.0 / float(np.sum(probs * probs))
        effective_neighbors[i] = int(labels.size)
    return scores, effective_neighbors, n_groups


def summarize_scores(
    scores: np.ndarray,
    *,
    group_col: str,
    group_role: str,
    metric_name: str,
    space: str,
    embed_source: str,
    embed_dim: int,
    k: int,
    n_spots_total: int,
    n_groups: int,
) -> dict[str, object]:
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        return {
            'group_col': group_col,
            'group_role': group_role,
            'metric_name': metric_name,
            'space': space,
            'embed_source': embed_source,
            'embed_dim': int(embed_dim),
            'k': int(k),
            'n_spots_total': int(n_spots_total),
            'n_valid': 0,
            'n_groups': int(n_groups),
            'lisi_mean': np.nan,
            'lisi_median': np.nan,
            'lisi_q25': np.nan,
            'lisi_q75': np.nan,
        }
    return {
        'group_col': group_col,
        'group_role': group_role,
        'metric_name': metric_name,
        'space': space,
        'embed_source': embed_source,
        'embed_dim': int(embed_dim),
        'k': int(k),
        'n_spots_total': int(n_spots_total),
        'n_valid': int(valid.size),
        'n_groups': int(n_groups),
        'lisi_mean': float(np.mean(valid)),
        'lisi_median': float(np.median(valid)),
        'lisi_q25': float(np.quantile(valid, 0.25)),
        'lisi_q75': float(np.quantile(valid, 0.75)),
    }
