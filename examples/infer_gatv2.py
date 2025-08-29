import argparse
import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data as PyGData

from preprocessing import Preprocessor, GraphBuilder
from utils import load_json
from model_gatv2 import SpotoncoGATv2Classifier


def assemble_pyg(Xp: np.ndarray, xy: np.ndarray, cfg: dict) -> PyGData:
    gb = GraphBuilder(knn_k=cfg['knn_k'], gaussian_sigma_factor=cfg['gaussian_sigma_factor'])
    edge_index, edge_weight, _ = gb.build_knn(xy)
    # 可选的Laplacian PE（支持是否拼接与是否使用高斯权重）
    if cfg.get('lap_pe_dim', 0) and cfg.get('lap_pe_dim', 0) > 0:
        pe = gb.lap_pe(
            edge_index,
            Xp.shape[0],
            k=cfg['lap_pe_dim'],
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


def infer_single(npz_path: str, artifacts_dir: str, out_csv: str, threshold: float = 0.5, device: str | None = None):
    # 读取 meta.json 获得 cfg，并设置默认值以保持兼容
    meta = load_json(os.path.join(artifacts_dir, 'meta.json'))
    cfg = dict(meta.get('cfg', {}))
    cfg.setdefault('lap_pe_dim', 16)
    cfg.setdefault('concat_lap_pe', True)
    cfg.setdefault('lap_pe_use_gaussian', False)
    cfg.setdefault('knn_k', 6)
    cfg.setdefault('gaussian_sigma_factor', 1.0)
    cfg.setdefault('heads', 4)
    cfg.setdefault('hidden', 128)
    cfg.setdefault('num_layers', 3)
    cfg.setdefault('dropout', 0.3)

    # 加载预处理器
    pp = Preprocessor.load(artifacts_dir)

    # 读取单NPZ（简单模式）
    d = np.load(npz_path, allow_pickle=True)
    if 'X' not in d or 'xy' not in d or 'gene_names' not in d:
        raise ValueError('NPZ 应包含键：X, xy, gene_names')
    X, xy, gene_names = d['X'], d['xy'], list(d['gene_names'])
    barcodes = d.get('barcodes', None)
    sample_id = str(d.get('sample_id', 'unknown'))

    # 预处理与构图
    Xp = pp.transform(X, gene_names)
    g = assemble_pyg(Xp, xy, cfg)

    # 设备
    dev = torch.device(device) if isinstance(device, str) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # 构建模型并加载权重（宽松加载，忽略不存在的域头参数）
    in_dim = g.x.shape[1]
    model = SpotoncoGATv2Classifier(
        in_dim=in_dim,
        hidden=cfg['hidden'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        heads=cfg['heads'],
        use_domain_adv_slide=False,
        use_domain_adv_cancer=False,
    ).to(dev)

    ckpt_path = os.path.join(artifacts_dir, 'model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'权重文件不存在: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=dev)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Load] Missing keys: {len(missing)} (ignored)")
    if unexpected:
        print(f"[Load] Unexpected keys: {len(unexpected)} (ignored)")

    model.eval()
    with torch.no_grad():
        g = g.to(dev)
        out = model(g.x, g.edge_index, batch=getattr(g, 'batch', None))
        probs = torch.sigmoid(out['logits']).detach().cpu().numpy()

    # 组织并保存结果
    n = probs.shape[0]
    if barcodes is None or len(barcodes) != n:
        barcodes_out = [f'spot_{i}' for i in range(n)]
    else:
        barcodes_out = list(map(str, barcodes))
    df = pd.DataFrame({
        'sample_id': sample_id,
        'Barcode': barcodes_out,
        'spot_idx': np.arange(n, dtype=int),
        'x': xy[:, 0].astype(float),
        'y': xy[:, 1].astype(float),
        'p_tumor': probs.astype(float),
    })
    df['pred_label'] = (df['p_tumor'] >= float(threshold)).astype(int)
    df['threshold'] = float(threshold)

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'Saved predictions to: {out_csv}  (n_spots={n})')


def main():
    p = argparse.ArgumentParser(description='Simple inference for GATv2 model (single NPZ)')
    p.add_argument('--npz', required=True, help='单切片NPZ，包含 X, xy, gene_names')
    p.add_argument('--artifacts_dir', default='artifacts', help='包含预处理与模型的目录')
    p.add_argument('--out_csv', default='preds.csv', help='预测CSV输出路径')
    p.add_argument('--threshold', type=float, default=0.5, help='二值阈值（仅用于输出pred_label）')
    p.add_argument('--device', default=None, help='设备：cuda 或 cpu；默认自动')
    args = p.parse_args()

    infer_single(args.npz, args.artifacts_dir, args.out_csv, threshold=args.threshold, device=args.device)


if __name__ == '__main__':
    main()