import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as PyGData

from stonco.core.models import STOnco_Classifier
from stonco.utils.preprocessing import GraphBuilder, Preprocessor
from stonco.utils.utils import load_json, load_model_state_dict


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


def _assemble_pyg(Xp: np.ndarray, xy: np.ndarray, cfg: dict) -> PyGData:
    gb = GraphBuilder(knn_k=int(cfg['knn_k']), gaussian_sigma_factor=float(cfg['gaussian_sigma_factor']))
    edge_index, edge_weight, _ = gb.build_knn(xy)
    if cfg.get('lap_pe_dim', 0) and int(cfg.get('lap_pe_dim', 0)) > 0:
        pe = gb.lap_pe(
            edge_index,
            Xp.shape[0],
            k=int(cfg['lap_pe_dim']),
            edge_weight=edge_weight if cfg.get('lap_pe_use_gaussian', False) else None,
            use_gaussian_weights=cfg.get('lap_pe_use_gaussian', False),
        )
    else:
        pe = None
    if cfg.get('concat_lap_pe', True) and pe is not None:
        x = np.hstack([Xp, pe]).astype('float32')
    else:
        x = Xp.astype('float32')
    data = PyGData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_weight=torch.from_numpy(edge_weight).float(),
    )
    data.num_nodes = x.shape[0]
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
        yield {
            'sample_id': sid,
            'X': Xs[i],
            'xy': xys[i],
            'y': ys[i],
            'gene_names': gene_names,
            'barcodes': barcodes,
        }


def _iter_samples_from_npz_glob(npz_glob: str):
    files = sorted(glob.glob(npz_glob))
    if not files:
        raise FileNotFoundError(f'No NPZ files found matching: {npz_glob}')
    for path in files:
        d = np.load(path, allow_pickle=True)
        keys = set(d.files)
        if not {'X', 'xy', 'gene_names'}.issubset(keys):
            print(f"[Warn] Skip {path}: require keys X, xy, gene_names. Found: {sorted(keys)}")
            continue
        sample_id = str(d.get('sample_id', Path(path).stem))
        y = d['y'] if 'y' in keys else None
        barcodes = d.get('barcodes', None)
        yield {
            'sample_id': sample_id,
            'X': d['X'],
            'xy': d['xy'],
            'y': y,
            'gene_names': list(d['gene_names']),
            'barcodes': barcodes,
        }


def main():
    parser = argparse.ArgumentParser(description='Export spot-level 64-d embeddings (z64) from a trained STOnco model.')
    parser.add_argument('--artifacts_dir', required=True, help='Training artifacts dir (contains meta.json, model weights, preprocessor).')
    parser.add_argument('--out_csv', default='spot_embeddings.csv', help='Output CSV path.')
    parser.add_argument('--append', action='store_true', help='Append to out_csv if it exists (default: overwrite).')
    parser.add_argument('--train_npz', default=None, help='Multi-slide NPZ with Xs/xys/ys/slide_ids/gene_names (and optional barcodes).')
    parser.add_argument('--npz_glob', default=None, help='Glob for single-slide NPZ files with X/xy/gene_names (optional y/sample_id/barcodes).')
    parser.add_argument('--subset', choices=['all', 'train', 'val'], default='all', help='When using --train_npz, filter by meta.json train/val ids.')
    parser.add_argument('--device', default=None, help='cpu/cuda; default auto.')
    parser.add_argument('--num_threads', type=int, default=None, help='Set torch CPU threads (only affects CPU).')
    args = parser.parse_args()

    mode_count = sum([args.train_npz is not None, args.npz_glob is not None])
    if mode_count != 1:
        raise ValueError('Please specify exactly one of --train_npz or --npz_glob')

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
    cfg.setdefault('concat_lap_pe', True)
    cfg.setdefault('lap_pe_use_gaussian', False)
    cfg.setdefault('knn_k', 6)
    cfg.setdefault('gaussian_sigma_factor', 1.0)
    cfg.setdefault('model', 'gatv2')
    cfg.setdefault('heads', 4)
    cfg.setdefault('hidden', 128)
    cfg.setdefault('num_layers', 3)
    cfg.setdefault('dropout', 0.3)
    cfg.setdefault('clf_hidden', [256, 128, 64])

    pp = Preprocessor.load(artifacts_dir)
    id2batch, id2type = _load_domain_maps()

    if args.train_npz is not None:
        sample_iter = _iter_samples_from_train_npz(args.train_npz, args.subset, meta)
    else:
        if args.subset != 'all':
            print('[Warn] --subset is only supported with --train_npz; ignored.')
        sample_iter = _iter_samples_from_npz_glob(args.npz_glob)

    model = None
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    if not args.append and os.path.exists(out_csv):
        os.remove(out_csv)
        print(f'[Info] Removed existing out_csv (overwrite): {out_csv}')

    header_written = bool(args.append and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0)

    for sample in sample_iter:
        sample_id = str(sample['sample_id'])
        X = sample['X']
        xy = sample['xy']
        y = sample.get('y', None)
        gene_names = list(sample['gene_names'])
        barcodes = sample.get('barcodes', None)

        Xp = pp.transform(X, gene_names)
        g = _assemble_pyg(Xp, xy, cfg).to(device)

        if model is None:
            in_dim = int(g.x.shape[1])
            clf_hidden = cfg.get('clf_hidden', [256, 128, 64])
            if isinstance(clf_hidden, str):
                clf_hidden = [int(x.strip()) for x in clf_hidden.split(',') if x.strip() != '']
            clf_hidden = [int(x) for x in clf_hidden]
            m = STOnco_Classifier(
                in_dim=in_dim,
                hidden=int(cfg['hidden']),
                num_layers=int(cfg['num_layers']),
                dropout=float(cfg['dropout']),
                model=str(cfg['model']),
                heads=int(cfg.get('heads', 4)),
                clf_hidden=clf_hidden,
            )
            _ = m.load_state_dict(load_model_state_dict(artifacts_dir, map_location=device), strict=False)
            model = m.to(device)
            model.eval()

        with torch.no_grad():
            out = model(
                g.x,
                g.edge_index,
                batch=getattr(g, 'batch', None),
                edge_weight=getattr(g, 'edge_weight', None),
                return_z=True,
            )
            logits = out['logits'].detach().cpu().numpy()
            z64 = out['z64'].detach().cpu().numpy()
            p_tumor = 1.0 / (1.0 + np.exp(-logits))

        if z64.ndim != 2 or z64.shape[1] != 64:
            raise ValueError(f'Expected z64 shape [N,64], got {z64.shape} for sample {sample_id}')

        n_spots = int(z64.shape[0])
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
        })
        for j in range(64):
            df[f'z64_{j}'] = z64[:, j]

        df.to_csv(out_csv, mode='a', header=(not header_written), index=False, float_format='%.6f')
        header_written = True
        print(f'[OK] {sample_id}: spots={n_spots}')

    print('Saved spot embeddings to', out_csv)


if __name__ == '__main__':
    main()
