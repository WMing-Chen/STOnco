import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stonco.utils.preprocessing import build_node_feature_fields, get_image_fusion_mode
from stonco.utils.utils import load_json, normalize_gnn_config, normalize_gnn_hidden
from stonco.core.model_utils import build_stonco_model_from_cfg, graph_input_dims, load_model_strict, model_forward_kwargs


def _resolve_batch_id(sample_id: str, id2batch: dict) -> str:
    bid = id2batch.get(sample_id)
    if bid is None or str(bid).strip() == '' or str(bid).lower() == 'nan':
        return sample_id
    return str(bid)


def _resolve_cancer_type(sample_id: str, id2type: dict) -> str:
    ctype = id2type.get(sample_id, None)
    if ctype is None or str(ctype).strip() == '' or str(ctype).lower() == 'nan':
        prefix = ''.join([ch for ch in sample_id if ch.isalpha()])
        return prefix if prefix else 'UNKNOWN'
    return str(ctype)


def _load_domain_maps():
    csv_path = Path(__file__).resolve().parents[2] / 'data' / 'cancer_sample_labels.csv'
    if not csv_path.exists():
        return {}, {}
    df = pd.read_csv(csv_path)
    id2type = {str(r['sample_id']): str(r['cancer_type']) for _, r in df.iterrows() if 'sample_id' in df.columns and 'cancer_type' in df.columns}
    id2batch = {}
    if 'Batch_id' in df.columns:
        id2batch = {str(r['sample_id']): str(r['Batch_id']) for _, r in df.iterrows() if 'sample_id' in df.columns}
    return id2batch, id2type


def _assemble_pyg(Xp_gene: np.ndarray, xy: np.ndarray, cfg: dict, Xp_img: np.ndarray | None = None, img_mask: np.ndarray | None = None):
    import torch
    from torch_geometric.data import Data as PyGData

    from stonco.utils.preprocessing import GraphBuilder

    gb = GraphBuilder(knn_k=int(cfg['knn_k']), gaussian_sigma_factor=float(cfg['gaussian_sigma_factor']))
    edge_index, edge_weight, _ = gb.build_knn(xy)
    if cfg.get('lap_pe_dim', 0) and int(cfg.get('lap_pe_dim', 0)) > 0:
        pe = gb.lap_pe(
            edge_index,
            Xp_gene.shape[0],
            k=int(cfg['lap_pe_dim']),
            edge_weight=edge_weight if cfg.get('lap_pe_use_gaussian', False) else None,
            use_gaussian_weights=cfg.get('lap_pe_use_gaussian', False),
        )
    else:
        pe = None
    fields = build_node_feature_fields(Xp_gene, cfg, Xp_img=Xp_img, img_mask=img_mask, pe=pe)
    data = PyGData(
        x=torch.from_numpy(fields['x']).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_weight=torch.from_numpy(edge_weight).float(),
    )
    if 'x_gene' in fields:
        data.x_gene = torch.from_numpy(fields['x_gene']).float()
    if 'x_img' in fields:
        data.x_img = torch.from_numpy(fields['x_img']).float()
    if 'img_mask' in fields:
        data.img_mask = torch.from_numpy(fields['img_mask']).float()
    if 'pe_gene' in fields:
        data.pe_gene = torch.from_numpy(fields['pe_gene']).float()
    data.num_nodes = fields['x'].shape[0]
    return data


def _iter_samples_from_train_npz(train_npz: str, subset: str, meta: dict):
    data = np.load(train_npz, allow_pickle=True)
    files = set(data.files)
    if not {'Xs', 'xys', 'ys', 'slide_ids', 'gene_names'}.issubset(files):
        raise ValueError(f"train_npz must contain Xs, xys, ys, slide_ids, gene_names. Found: {sorted(files)}")

    Xs = list(data['Xs'])
    xys = list(data['xys'])
    ys = list(data['ys'])
    slide_ids = [str(s) for s in list(data['slide_ids'])]
    gene_names = list(data['gene_names'])
    barcodes_list = list(data['barcodes']) if 'barcodes' in files else None
    X_imgs = list(data['X_imgs']) if 'X_imgs' in files else None
    img_masks = list(data['img_masks']) if 'img_masks' in files else None
    img_feature_names = list(data['img_feature_names']) if 'img_feature_names' in files else None

    keep_ids = None
    if subset != 'all':
        meta_ids = meta.get(f'{subset}_ids', None)
        if isinstance(meta_ids, list) and meta_ids:
            keep_ids = set(map(str, meta_ids))

    for i, sid in enumerate(slide_ids):
        if keep_ids is not None and sid not in keep_ids:
            continue
        barcodes = None
        if barcodes_list is not None:
            try:
                barcodes = barcodes_list[i]
            except Exception:
                barcodes = None
        X_img = None
        img_mask = None
        if X_imgs is not None and img_masks is not None:
            try:
                X_img = X_imgs[i]
                img_mask = img_masks[i]
            except Exception:
                X_img = None
                img_mask = None
        yield {
            'sample_id': sid,
            'X': Xs[i],
            'xy': xys[i],
            'y': ys[i],
            'gene_names': gene_names,
            'barcodes': barcodes,
            'X_img': X_img,
            'img_mask': img_mask,
            'img_feature_names': img_feature_names,
        }


def _iter_samples_from_npz_files(files: list[str], source_desc: str):
    if not files:
        raise FileNotFoundError(f'No NPZ files found for: {source_desc}')
    for path in files:
        d = np.load(path, allow_pickle=True)
        keys = set(d.files)
        if not {'X', 'xy', 'gene_names'}.issubset(keys):
            print(f"[Warn] Skip {path}: require keys X, xy, gene_names. Found: {sorted(keys)}")
            continue
        sample_id = str(d.get('sample_id', Path(path).stem))
        y = d['y'] if 'y' in keys else None
        barcodes = d.get('barcodes', None)
        X_img = d['X_img'] if 'X_img' in keys else None
        img_mask = d['img_mask'] if 'img_mask' in keys else None
        img_feature_names = list(d['img_feature_names']) if 'img_feature_names' in keys else None
        yield {
            'sample_id': sample_id,
            'X': d['X'],
            'xy': d['xy'],
            'y': y,
            'gene_names': list(d['gene_names']),
            'barcodes': barcodes,
            'X_img': X_img,
            'img_mask': img_mask,
            'img_feature_names': img_feature_names,
        }


def _iter_samples_from_npz_glob(npz_glob: str):
    files = sorted(glob.glob(npz_glob))
    yield from _iter_samples_from_npz_files(files, npz_glob)


def _iter_samples_from_npz_paths(npz_paths: list[str]):
    files = [os.path.abspath(path) for path in npz_paths]
    yield from _iter_samples_from_npz_files(files, ', '.join(files))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export spot-level embeddings (h or classifier latent) from a trained STOnco model.')
    parser.add_argument('--artifacts_dir', required=True, help='Training artifacts dir (contains meta.json, model weights, preprocessor).')
    parser.add_argument('--out_csv', default='spot_embeddings.csv', help='Output CSV path.')
    parser.add_argument('--append', action='store_true', help='Append to out_csv if it exists (default: overwrite).')
    parser.add_argument('--train_npz', default=None, help='Multi-slide NPZ with Xs/xys/ys/slide_ids/gene_names (and optional barcodes).')
    parser.add_argument('--npz', action='append', default=None, help='Single-slide NPZ path. Can be repeated.')
    parser.add_argument('--npz_glob', action='append', default=None, help='Glob for single-slide NPZ files with X/xy/gene_names (optional y/sample_id/barcodes). Can be repeated.')
    parser.add_argument('--subset', choices=['all', 'train', 'val'], default='all', help='When using --train_npz, filter by meta.json train/val ids. Ignored for --npz/--npz_glob.')
    parser.add_argument('--device', default=None, help='cpu/cuda; default auto.')
    parser.add_argument('--num_threads', type=int, default=None, help='Set torch CPU threads (only affects CPU).')
    parser.add_argument(
        '--embed_source',
        choices=['h', 'z_clf', 'z64'],
        default='h',
        help='Embedding source for export: h (GNN output), z_clf (classifier latent), or z64 (legacy alias for 64-dim classifier latent).',
    )
    return parser


def run_export(args: argparse.Namespace) -> str:
    from stonco.utils.preprocessing import ImagePreprocessor, Preprocessor
    import torch

    has_any_input = any([
        args.train_npz is not None,
        bool(args.npz),
        bool(args.npz_glob),
    ])
    if not has_any_input:
        raise ValueError('Please provide at least one input source: --train_npz and/or --npz and/or --npz_glob')

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cpu' and args.num_threads is not None and args.num_threads > 0:
        try:
            torch.set_num_threads(args.num_threads)
            torch.set_num_interop_threads(max(1, min(args.num_threads, 2)))
        except Exception as e:
            print(f"[Warn] set threads failed: {e}")

    artifacts_dir = os.path.abspath(args.artifacts_dir)
    meta = load_json(os.path.join(artifacts_dir, 'meta.json'))
    cfg = dict(meta.get('cfg', {}))
    cfg.setdefault('lap_pe_dim', 16)
    cfg.setdefault('concat_lap_pe', False)
    cfg.setdefault('lap_pe_use_gaussian', False)
    cfg.setdefault('knn_k', 6)
    cfg.setdefault('gaussian_sigma_factor', 1.0)
    cfg.setdefault('model', 'gatv2')
    cfg.setdefault('heads', 4)
    cfg.setdefault('num_layers', 3)
    cfg.setdefault('dropout', 0.3)
    cfg['gnn_dropout'] = float(cfg['dropout'] if cfg.get('gnn_dropout', None) is None else cfg['gnn_dropout'])
    cfg['clf_dropout'] = float(cfg['dropout'] if cfg.get('clf_dropout', None) is None else cfg['clf_dropout'])
    cfg['dom_dropout'] = float(cfg['dropout'] if cfg.get('dom_dropout', None) is None else cfg['dom_dropout'])
    cfg.setdefault('clf_hidden', [256, 128, 64])
    cfg.setdefault('use_image_features', False)
    cfg.setdefault('image_fusion_mode', 'early_concat')
    cfg.setdefault('img_use_pca', False)
    cfg.setdefault('img_pca_dim', 256)
    cfg.setdefault('img_gnn_hidden', [128, 64])
    cfg.setdefault('img_num_layers', 2)
    cfg.setdefault('img_model', cfg.get('model', 'gatv2'))
    cfg.setdefault('img_heads', cfg.get('heads', 4))
    cfg.setdefault('img_dropout', cfg.get('gnn_dropout', cfg.get('dropout', 0.3)))
    cfg.setdefault('fusion_gate_hidden', 128)
    cfg.setdefault('fusion_dropout', 0.0)
    clf_hidden = cfg.get('clf_hidden', [256, 128, 64])
    if isinstance(clf_hidden, str):
        clf_hidden = [int(x.strip()) for x in clf_hidden.split(',') if x.strip() != '']
    cfg['clf_hidden'] = [int(x) for x in clf_hidden]
    cfg.setdefault('clf_latent_dim', int(cfg['clf_hidden'][-1]))
    cfg = normalize_gnn_config(cfg)
    cfg['image_fusion_mode'] = get_image_fusion_mode(cfg)
    img_hidden, img_layers = normalize_gnn_hidden(
        gnn_hidden=cfg.get('img_gnn_hidden', [128, 64]),
        num_layers=cfg.get('img_num_layers', 2),
        default_gnn_hidden=(128, 64),
    )
    cfg['img_gnn_hidden'] = [int(v) for v in img_hidden]
    cfg['img_num_layers'] = int(img_layers)

    pp = Preprocessor.load(artifacts_dir)
    use_image_features = bool(cfg.get('use_image_features', False))
    img_pp = None
    expected_img_feature_names = None
    if use_image_features:
        img_pp = ImagePreprocessor.load(artifacts_dir, img_use_pca=bool(cfg.get('img_use_pca', False)))
        expected_img_feature_names = img_pp.feature_names
        if not expected_img_feature_names:
            raise ValueError('use_image_features=1 but artifacts_dir/img_feature_names.txt is missing or empty.')
    id2batch, id2type = _load_domain_maps()

    model = None
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    if not args.append and os.path.exists(out_csv):
        os.remove(out_csv)
        print(f'[Info] Removed existing out_csv (overwrite): {out_csv}')

    header_written = bool(args.append and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0)

    input_iters = []
    if args.train_npz is not None:
        input_iters.append(_iter_samples_from_train_npz(args.train_npz, args.subset, meta))
    elif args.subset != 'all':
        print('[Warn] --subset is only supported with --train_npz; ignored.')

    if args.npz:
        input_iters.append(_iter_samples_from_npz_paths(args.npz))
    if args.npz_glob:
        for npz_glob in args.npz_glob:
            input_iters.append(_iter_samples_from_npz_glob(npz_glob))

    for sample_iter in input_iters:
        for sample in sample_iter:
            sample_id = str(sample['sample_id'])
            X = sample['X']
            xy = sample['xy']
            y = sample.get('y', None)
            gene_names = list(sample['gene_names'])
            barcodes = sample.get('barcodes', None)

            Xp_gene = pp.transform(X, gene_names)
            if use_image_features:
                X_img = sample.get('X_img', None)
                img_mask = sample.get('img_mask', None)
                has_img_arrays = X_img is not None and img_mask is not None
                if has_img_arrays:
                    img_feature_names = sample.get('img_feature_names', None)
                    if img_feature_names is None:
                        raise ValueError(f"Sample {sample_id} contains X_img but missing img_feature_names")
                    if expected_img_feature_names is not None and list(img_feature_names) != expected_img_feature_names:
                        raise ValueError(f'img_feature_names mismatch for sample {sample_id}')
                else:
                    X_img = np.zeros((X.shape[0], int(img_pp.scaler.mean_.shape[0])), dtype=np.float32)
                    img_mask = np.zeros((X.shape[0],), dtype=np.uint8)
                Xp_img = img_pp.transform(X_img, img_mask)
                g = _assemble_pyg(Xp_gene, xy, cfg, Xp_img=Xp_img, img_mask=img_mask).to(device)
            else:
                g = _assemble_pyg(Xp_gene, xy, cfg).to(device)

            if model is None:
                clf_hidden = cfg.get('clf_hidden', [256, 128, 64])
                cfg['clf_latent_dim'] = int(clf_hidden[-1])
                m = build_stonco_model_from_cfg(graph_input_dims(g), cfg)
                load_model_strict(m, artifacts_dir, map_location=device)
                model = m.to(device)
                model.eval()

            with torch.no_grad():
                need_z = (args.embed_source in ('z_clf', 'z64'))
                need_h = (args.embed_source == 'h')
                out = model(
                    **model_forward_kwargs(g),
                    return_z=need_z,
                    return_h=need_h,
                )
                logits = out['logits'].detach().cpu().numpy()
                p_tumor = 1.0 / (1.0 + np.exp(-logits))
                if args.embed_source in ('z_clf', 'z64'):
                    emb = out['z_clf'].detach().cpu().numpy()
                    emb_prefix = 'z_clf_'
                    if args.embed_source == 'z64' and int(emb.shape[1]) != 64:
                        raise ValueError(
                            f'--embed_source z64 requires classifier latent dim == 64, got {emb.shape[1]}. '
                            'Use --embed_source z_clf for dynamically sized classifier embeddings.'
                        )
                else:
                    emb = out['h'].detach().cpu().numpy()
                    emb_prefix = 'h_'

            if emb.ndim != 2:
                raise ValueError(f'Expected embedding shape [N,D], got {emb.shape} for sample {sample_id}')
            n_spots = int(emb.shape[0])
            emb_dim = int(emb.shape[1])
            if emb_dim < 2:
                raise ValueError(f'Embedding dim must be >=2, got {emb_dim} for sample {sample_id}')

            if barcodes is not None:
                try:
                    barcodes = np.array(barcodes).astype(str)
                    if barcodes.shape[0] != n_spots:
                        barcodes = None
                except Exception:
                    barcodes = None
            spot_id = barcodes if barcodes is not None else np.array([f'{sample_id}:{i}' for i in range(n_spots)], dtype=object)

            batch_id = _resolve_batch_id(sample_id, id2batch)
            cancer_type = _resolve_cancer_type(sample_id, id2type)

            df = pd.DataFrame({
                'spot_id': spot_id,
                'sample_id': [sample_id] * n_spots,
                'spot_idx': np.arange(n_spots, dtype=int),
                'x': np.asarray(xy)[:, 0],
                'y': np.asarray(xy)[:, 1],
                'tumor_label': (np.asarray(y).astype(int) if y is not None else np.full(n_spots, np.nan)),
                'batch_id': [batch_id] * n_spots,
                'cancer_type': [cancer_type] * n_spots,
                'p_tumor': p_tumor,
                'embed_source': [('z_clf' if args.embed_source in ('z_clf', 'z64') else 'h')] * n_spots,
                'embed_dim': [emb_dim] * n_spots,
                'clf_latent_dim': [int(cfg.get('clf_latent_dim', emb_dim))] * n_spots,
            })
            emb_cols = [f'{emb_prefix}{j}' for j in range(emb_dim)]
            emb_df = pd.DataFrame(emb, columns=emb_cols)
            df = pd.concat([df, emb_df], axis=1)

            df.to_csv(out_csv, mode='a', header=(not header_written), index=False, float_format='%.6f')
            header_written = True
            print(f'[OK] {sample_id}: spots={n_spots}')

    print('Saved spot embeddings to', out_csv)
    return out_csv


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_export(args)


if __name__ == '__main__':
    main()
