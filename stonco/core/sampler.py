import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import BatchSampler
from torch_geometric.data import Data as PyGData

from stonco.utils.preprocessing import GraphBuilder


def infer_default_sampler_structure(batch_size):
    batch_size = int(batch_size)
    if batch_size < 4:
        return 1, batch_size
    if batch_size == 4:
        return 2, 2
    if batch_size == 6:
        return 3, 2
    if batch_size == 8:
        return 4, 2
    if batch_size >= 12 and batch_size % 3 == 0:
        return max(1, batch_size // 3), 3
    if batch_size >= 8:
        return max(1, batch_size // 2), 2
    return max(1, batch_size // 2), 2


def normalize_sampler_config(cfg):
    sampler_mode = str(cfg.get('sampler_mode', 'random')).lower()
    if sampler_mode == 'cancer_balanced_subgraph':
        sampler_mode = 'cancer_balanced'
        cfg['subgraph_mode'] = 'static'
    if sampler_mode not in {'random', 'cancer_balanced'}:
        raise ValueError(
            "cfg['sampler_mode'] must be one of random/cancer_balanced/cancer_balanced_subgraph, "
            f"got: {cfg.get('sampler_mode')}"
        )

    subgraph_mode = str(cfg.get('subgraph_mode', 'off')).lower()
    if subgraph_mode not in {'off', 'static', 'online'}:
        raise ValueError(f"cfg['subgraph_mode'] must be one of off/static/online, got: {cfg.get('subgraph_mode')}")

    cfg['sampler_mode'] = sampler_mode
    cfg['subgraph_mode'] = subgraph_mode
    cfg['sampler_enforce_distinct_batch'] = bool(cfg.get('sampler_enforce_distinct_batch', True))
    cfg['sampler_seed'] = int(cfg.get('sampler_seed', 42))
    cfg['subgraph_target_spots'] = int(cfg.get('subgraph_target_spots', 1000))
    cfg['subgraph_min_spots'] = int(cfg.get('subgraph_min_spots', 300))

    batch_size = int(cfg.get('batch_size_graphs', 1))
    if cfg.get('sampler_k_cancers', None) is None or cfg.get('sampler_m_per_cancer', None) is None:
        k_default, m_default = infer_default_sampler_structure(batch_size)
        if cfg.get('sampler_k_cancers', None) is None:
            cfg['sampler_k_cancers'] = int(k_default)
        if cfg.get('sampler_m_per_cancer', None) is None:
            cfg['sampler_m_per_cancer'] = int(m_default)
    else:
        cfg['sampler_k_cancers'] = int(cfg['sampler_k_cancers'])
        cfg['sampler_m_per_cancer'] = int(cfg['sampler_m_per_cancer'])

    if int(cfg['sampler_k_cancers']) <= 0 or int(cfg['sampler_m_per_cancer']) <= 0:
        raise ValueError(
            f"cfg['sampler_k_cancers'] and cfg['sampler_m_per_cancer'] must be positive, got: "
            f"{cfg['sampler_k_cancers']}, {cfg['sampler_m_per_cancer']}"
        )
    return cfg


def _graph_attr(graph, name, default=None):
    if hasattr(graph, name):
        value = getattr(graph, name)
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.item()
            return value
        return value
    return default


def _graph_parent_id(graph):
    return str(_graph_attr(graph, 'parent_slide_id', _graph_attr(graph, 'slide_id')))


def _graph_slide_id(graph):
    return str(_graph_attr(graph, 'slide_id'))


def _graph_batch_id(graph):
    value = _graph_attr(graph, 'bat_dom', None)
    if value is None:
        return None
    return int(value)


def _graph_cancer_id(graph):
    value = _graph_attr(graph, 'cancer_dom', None)
    if value is None:
        return None
    return int(value)


class CancerBalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        graphs,
        batch_size,
        k_cancers,
        m_per_cancer,
        seed=42,
        enforce_distinct_batch=True,
        num_batches=None,
    ):
        self.graphs = list(graphs)
        self.batch_size = int(batch_size)
        self.k_cancers = int(k_cancers)
        self.m_per_cancer = int(m_per_cancer)
        self.seed = int(seed)
        self.enforce_distinct_batch = bool(enforce_distinct_batch)
        self.num_batches = int(num_batches) if num_batches is not None else int(math.ceil(len(self.graphs) / float(max(self.batch_size, 1))))
        self._epoch = 0

        self.meta = []
        self.by_cancer = defaultdict(list)
        for idx, graph in enumerate(self.graphs):
            cancer_id = _graph_cancer_id(graph)
            if cancer_id is None:
                raise ValueError('CancerBalancedBatchSampler requires each training graph to have cancer_dom.')
            meta = {
                'index': idx,
                'cancer_dom': int(cancer_id),
                'bat_dom': _graph_batch_id(graph),
                'slide_id': _graph_slide_id(graph),
                'parent_slide_id': _graph_parent_id(graph),
            }
            self.meta.append(meta)
            self.by_cancer[int(cancer_id)].append(meta)

        self.cancers = sorted(self.by_cancer.keys())
        if not self.cancers:
            raise ValueError('CancerBalancedBatchSampler received no training graphs.')

    def __len__(self):
        return self.num_batches

    def _choose_cancers(self, rng):
        if len(self.cancers) <= self.k_cancers:
            return list(self.cancers)
        return rng.sample(self.cancers, self.k_cancers)

    def _sample_from_cancer(self, cancer_id, rng, used_parent_ids):
        candidates = list(self.by_cancer.get(int(cancer_id), []))
        rng.shuffle(candidates)
        selected = []
        used_batch_ids = set()

        if self.enforce_distinct_batch:
            by_batch = defaultdict(list)
            for meta in candidates:
                by_batch[meta['bat_dom']].append(meta)
            batch_ids = list(by_batch.keys())
            rng.shuffle(batch_ids)
            for batch_id in batch_ids:
                if len(selected) >= self.m_per_cancer:
                    break
                local = list(by_batch[batch_id])
                rng.shuffle(local)
                for meta in local:
                    if meta['parent_slide_id'] in used_parent_ids:
                        continue
                    selected.append(meta)
                    used_parent_ids.add(meta['parent_slide_id'])
                    used_batch_ids.add(batch_id)
                    break

        if len(selected) < self.m_per_cancer:
            for meta in candidates:
                if len(selected) >= self.m_per_cancer:
                    break
                if meta['parent_slide_id'] in used_parent_ids:
                    continue
                if self.enforce_distinct_batch and meta['bat_dom'] in used_batch_ids:
                    continue
                selected.append(meta)
                used_parent_ids.add(meta['parent_slide_id'])
                used_batch_ids.add(meta['bat_dom'])

        if len(selected) < self.m_per_cancer:
            for meta in candidates:
                if len(selected) >= self.m_per_cancer:
                    break
                if meta['parent_slide_id'] in used_parent_ids:
                    continue
                selected.append(meta)
                used_parent_ids.add(meta['parent_slide_id'])
                used_batch_ids.add(meta['bat_dom'])

        return selected

    def _fill_from_other_cancers(self, selected_cancers, used_parent_ids, rng, needed):
        if needed <= 0:
            return []

        remaining = []
        other_cancers = [c for c in self.cancers if c not in set(selected_cancers)]
        rng.shuffle(other_cancers)
        for cancer_id in other_cancers:
            local = list(self.by_cancer[cancer_id])
            rng.shuffle(local)
            for meta in local:
                if len(remaining) >= needed:
                    return remaining
                if meta['parent_slide_id'] in used_parent_ids:
                    continue
                remaining.append(meta)
                used_parent_ids.add(meta['parent_slide_id'])
        return remaining

    def _fill_with_replacement(self, rng, needed):
        if needed <= 0:
            return []
        flat = list(self.meta)
        if not flat:
            return []
        return [rng.choice(flat) for _ in range(needed)]

    def _build_batch(self, rng):
        selected_cancers = self._choose_cancers(rng)
        batch = []
        used_parent_ids = set()

        for cancer_id in selected_cancers:
            batch.extend(self._sample_from_cancer(cancer_id, rng, used_parent_ids))

        if len(batch) < self.batch_size:
            batch.extend(self._fill_from_other_cancers(selected_cancers, used_parent_ids, rng, self.batch_size - len(batch)))

        if len(batch) < self.batch_size:
            batch.extend(self._fill_with_replacement(rng, self.batch_size - len(batch)))

        rng.shuffle(batch)
        return [meta['index'] for meta in batch[:self.batch_size]]

    def _build_epoch_batches(self, epoch):
        rng = random.Random(self.seed + int(epoch) * 1009)
        return [self._build_batch(rng) for _ in range(self.num_batches)]

    def preview_batches(self):
        return self._build_epoch_batches(self._epoch)

    def __iter__(self):
        batches = self._build_epoch_batches(self._epoch)
        self._epoch += 1
        for batch in batches:
            yield batch


def summarize_batches(graphs, batches):
    if not batches:
        return {
            'avg_unique_cancers': float('nan'),
            'avg_unique_batches': float('nan'),
            'avg_unique_slides': float('nan'),
            'avg_unique_parents': float('nan'),
        }

    uniq_cancers = []
    uniq_batches = []
    uniq_slides = []
    uniq_parents = []
    for batch in batches:
        metas = [graphs[idx] for idx in batch]
        uniq_cancers.append(len({_graph_cancer_id(g) for g in metas}))
        uniq_batches.append(len({_graph_batch_id(g) for g in metas}))
        uniq_slides.append(len({_graph_slide_id(g) for g in metas}))
        uniq_parents.append(len({_graph_parent_id(g) for g in metas}))
    return {
        'avg_unique_cancers': float(np.mean(uniq_cancers)),
        'avg_unique_batches': float(np.mean(uniq_batches)),
        'avg_unique_slides': float(np.mean(uniq_slides)),
        'avg_unique_parents': float(np.mean(uniq_parents)),
    }


def build_training_subgraphs(train_graphs, cfg):
    subgraph_mode = str(cfg.get('subgraph_mode', 'off')).lower()
    if subgraph_mode == 'off':
        return list(train_graphs)
    if subgraph_mode == 'online':
        raise NotImplementedError("subgraph_mode='online' is reserved but not implemented yet. Use 'static' for now.")
    if subgraph_mode != 'static':
        raise ValueError(f"Unsupported subgraph_mode: {subgraph_mode}")

    knn_k = int(cfg.get('knn_k', 6))
    sigma_factor = float(cfg.get('gaussian_sigma_factor', 1.0))
    target_spots = int(cfg.get('subgraph_target_spots', 1000))
    min_spots = int(cfg.get('subgraph_min_spots', 300))

    all_subgraphs = []
    for graph in train_graphs:
        all_subgraphs.extend(
            split_graph_into_subgraphs(
                graph,
                knn_k=knn_k,
                gaussian_sigma_factor=sigma_factor,
                target_spots=target_spots,
                min_spots=min_spots,
            )
        )
    return all_subgraphs


def split_graph_into_subgraphs(graph, knn_k, gaussian_sigma_factor, target_spots=1000, min_spots=300):
    if not hasattr(graph, 'pos'):
        raise ValueError('Subgraph training requires each graph to carry spatial coordinates in graph.pos.')

    num_nodes = int(graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0))
    parent_slide_id = _graph_slide_id(graph)
    if num_nodes <= max(int(target_spots), int(min_spots)):
        whole = _clone_full_graph_as_subgraph(graph, parent_slide_id, 0)
        return [whole]

    coords = graph.pos.detach().cpu().numpy()
    partitions = _partition_spatial_indices(coords, target_spots=target_spots, min_spots=min_spots)
    if len(partitions) <= 1:
        whole = _clone_full_graph_as_subgraph(graph, parent_slide_id, 0)
        return [whole]

    builder = GraphBuilder(knn_k=int(knn_k), gaussian_sigma_factor=float(gaussian_sigma_factor))
    subgraphs = []
    for idx, node_idx in enumerate(partitions):
        node_idx = np.asarray(node_idx, dtype=np.int64)
        node_idx_t = torch.from_numpy(node_idx).long()
        coords_sub = coords[node_idx]
        edge_index, edge_weight, _ = builder.build_knn(coords_sub)
        sub = PyGData(
            x=graph.x[node_idx_t].clone(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_weight=torch.from_numpy(edge_weight).float(),
            y=graph.y[node_idx_t].clone(),
        )
        sub.num_nodes = int(node_idx_t.numel())
        sub.pos = torch.from_numpy(coords_sub).float()
        sub.slide_id = f"{parent_slide_id}__sg{idx:03d}"
        sub.parent_slide_id = str(parent_slide_id)
        sub.subgraph_id = f"sg{idx:03d}"
        if hasattr(graph, 'bat_dom'):
            sub.bat_dom = graph.bat_dom.clone()
        if hasattr(graph, 'cancer_dom'):
            sub.cancer_dom = graph.cancer_dom.clone()
        subgraphs.append(sub)
    return subgraphs


def _clone_full_graph_as_subgraph(graph, parent_slide_id, idx):
    cloned = PyGData(
        x=graph.x.clone(),
        edge_index=graph.edge_index.clone(),
        edge_weight=graph.edge_weight.clone() if hasattr(graph, 'edge_weight') and graph.edge_weight is not None else None,
        y=graph.y.clone(),
    )
    cloned.num_nodes = int(graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0))
    cloned.pos = graph.pos.clone()
    cloned.slide_id = f"{parent_slide_id}__sg{idx:03d}"
    cloned.parent_slide_id = str(parent_slide_id)
    cloned.subgraph_id = f"sg{idx:03d}"
    if hasattr(graph, 'bat_dom'):
        cloned.bat_dom = graph.bat_dom.clone()
    if hasattr(graph, 'cancer_dom'):
        cloned.cancer_dom = graph.cancer_dom.clone()
    return cloned


def _partition_spatial_indices(coords, target_spots=1000, min_spots=300):
    num_nodes = int(coords.shape[0])
    if num_nodes <= max(int(target_spots), int(min_spots)):
        return [np.arange(num_nodes, dtype=np.int64)]

    n_parts = max(1, int(math.ceil(num_nodes / float(max(int(target_spots), 1)))))
    nx = max(1, int(math.ceil(math.sqrt(n_parts))))
    ny = max(1, int(math.ceil(n_parts / float(nx))))

    x_bins = _make_bin_edges(coords[:, 0], nx)
    y_bins = _make_bin_edges(coords[:, 1], ny)
    x_idx = np.clip(np.searchsorted(x_bins, coords[:, 0], side='right') - 1, 0, len(x_bins) - 2)
    y_idx = np.clip(np.searchsorted(y_bins, coords[:, 1], side='right') - 1, 0, len(y_bins) - 2)

    cells = defaultdict(list)
    for i in range(num_nodes):
        cells[(int(x_idx[i]), int(y_idx[i]))].append(i)

    large_groups = []
    large_centers = []
    small_groups = []
    for key in sorted(cells.keys()):
        indices = np.asarray(cells[key], dtype=np.int64)
        if indices.size >= int(min_spots):
            large_groups.append(indices)
            large_centers.append(coords[indices].mean(axis=0))
        else:
            small_groups.append(indices)

    if not large_groups:
        return [np.arange(num_nodes, dtype=np.int64)]

    for indices in small_groups:
        center = coords[indices].mean(axis=0)
        nearest = int(np.argmin([np.sum((center - c) ** 2) for c in large_centers]))
        large_groups[nearest] = np.unique(np.concatenate([large_groups[nearest], indices])).astype(np.int64)
        large_centers[nearest] = coords[large_groups[nearest]].mean(axis=0)

    return [np.asarray(sorted(group.tolist()), dtype=np.int64) for group in large_groups if int(group.size) > 0]


def _make_bin_edges(values, n_bins):
    values = np.asarray(values, dtype=np.float64)
    if int(n_bins) <= 1 or values.size == 0:
        vmin = float(np.min(values)) if values.size else 0.0
        vmax = float(np.max(values)) if values.size else 1.0
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        return np.asarray([vmin, vmax + 1e-6], dtype=np.float64)

    quantiles = np.quantile(values, np.linspace(0.0, 1.0, int(n_bins) + 1))
    if np.unique(quantiles).size < int(n_bins) + 1:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        edges = np.linspace(vmin, vmax + 1e-6, int(n_bins) + 1)
    else:
        edges = np.asarray(quantiles, dtype=np.float64)
        edges[-1] = edges[-1] + 1e-6
    return edges
