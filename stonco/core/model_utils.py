from __future__ import annotations

import os

import torch

from stonco.utils.preprocessing import get_image_fusion_mode
from stonco.utils.utils import load_model_state_dict

from .models import STOnco_Classifier


def graph_input_dims(graph):
    gene_dim = int(graph.x_gene.shape[1]) if hasattr(graph, 'x_gene') else int(graph.x.shape[1])
    if hasattr(graph, 'pe_gene') and graph.pe_gene is not None:
        gene_dim += int(graph.pe_gene.shape[1])
    return {
        'gene_in_dim': gene_dim,
        'img_in_dim': int(graph.x_img.shape[1]) if hasattr(graph, 'x_img') else None,
        'x_in_dim': int(graph.x.shape[1]),
    }


def model_forward_kwargs(graph):
    kwargs = {
        'x': graph.x,
        'edge_index': graph.edge_index,
        'batch': getattr(graph, 'batch', None),
        'edge_weight': getattr(graph, 'edge_weight', None),
    }
    if hasattr(graph, 'x_gene'):
        kwargs['x_gene'] = graph.x_gene
    if hasattr(graph, 'x_img'):
        kwargs['x_img'] = graph.x_img
    if hasattr(graph, 'img_mask'):
        kwargs['img_mask'] = graph.img_mask
    if hasattr(graph, 'pe_gene'):
        kwargs['pe_gene'] = graph.pe_gene
    return kwargs


def build_stonco_model_from_cfg(input_dims, cfg, n_domains_slide=None, n_domains_cancer=None):
    dims = dict(input_dims) if isinstance(input_dims, dict) else {
        'gene_in_dim': int(input_dims),
        'img_in_dim': None,
        'x_in_dim': int(input_dims),
    }
    mode = get_image_fusion_mode(cfg)
    extra = {'image_fusion_mode': mode}
    if mode == 'dual_branch_residual_gate':
        img_num_layers = cfg.get('img_num_layers', 2)
        img_heads = cfg.get('img_heads', None)
        img_dropout = cfg.get('img_dropout', None)
        fusion_gate_hidden = cfg.get('fusion_gate_hidden', 128)
        fusion_dropout = cfg.get('fusion_dropout', 0.0)
        extra.update({
            'in_dim_img': dims.get('img_in_dim'),
            'img_hidden': cfg.get('img_gnn_hidden', [128, 64]),
            'img_num_layers': int(2 if img_num_layers is None else img_num_layers),
            'img_model': cfg.get('img_model', cfg.get('model', 'gatv2')),
            'img_heads': int(cfg.get('heads', 4) if img_heads is None else img_heads),
            'img_dropout': cfg.get('gnn_dropout', cfg.get('dropout', 0.3)) if img_dropout is None else img_dropout,
            'fusion_gate_hidden': int(128 if fusion_gate_hidden is None else fusion_gate_hidden),
            'fusion_dropout': float(0.0 if fusion_dropout is None else fusion_dropout),
        })
    in_dim = int(dims['gene_in_dim'] if mode == 'dual_branch_residual_gate' else dims.get('x_in_dim', dims['gene_in_dim']))
    return STOnco_Classifier(
        in_dim=in_dim,
        hidden=cfg['GNN_hidden'],
        num_layers=int(cfg['num_layers']),
        dropout=float(cfg['dropout']),
        gnn_dropout=float(cfg.get('gnn_dropout', cfg.get('dropout', 0.3))),
        clf_dropout=float(cfg.get('clf_dropout', cfg.get('dropout', 0.3))),
        dom_dropout=float(cfg.get('dom_dropout', cfg.get('dropout', 0.3))),
        model=str(cfg['model']),
        heads=int(cfg.get('heads', 4)),
        clf_hidden=cfg.get('clf_hidden', [256, 128, 64]),
        use_domain_adv_slide=cfg.get('use_domain_adv_slide', False),
        n_domains_slide=n_domains_slide,
        use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False),
        n_domains_cancer=n_domains_cancer,
        domain_hidden=int(cfg.get('dom_hidden', 64)),
        **extra,
    )


def load_model_strict(model, artifacts_dir, map_location='cpu'):
    try:
        model.load_state_dict(load_model_state_dict(artifacts_dir, map_location=map_location), strict=True)
    except RuntimeError as exc:
        mode = getattr(model, 'image_fusion_mode', 'unknown')
        ckpt_path = os.path.join(artifacts_dir, 'model.pt')
        raise RuntimeError(
            f'Failed to load checkpoint with strict=True for image_fusion_mode={mode}. '
            f'Checkpoint path: {ckpt_path}. Build the model with the same image_fusion_mode as meta.json.'
        ) from exc
    return model
