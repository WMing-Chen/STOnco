import argparse
import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data as PyGData, DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from preprocessing import Preprocessor, GraphBuilder
from model_gatv2 import SpotoncoGATv2Classifier
from utils import save_model, save_json


def assemble_pyg(Xp: np.ndarray, xy: np.ndarray, y: np.ndarray, cfg: dict) -> PyGData:
    gb = GraphBuilder(knn_k=cfg['knn_k'], gaussian_sigma_factor=cfg['gaussian_sigma_factor'])
    edge_index, edge_weight, _ = gb.build_knn(xy)
    # 可选 Laplacian 位置编码
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
        y=torch.from_numpy(y).long(),
    )
    data.num_nodes = x.shape[0]
    return data


def eval_logits(logits: torch.Tensor, y: torch.Tensor) -> dict:
    mask = y >= 0
    if mask.sum() == 0:
        return {"auroc": float('nan'), "auprc": float('nan'), "accuracy": float('nan'), "macro_f1": float('nan')}
    ytrue = y[mask].cpu().numpy()
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(ytrue, probs))
        auprc = float(average_precision_score(ytrue, probs))
        accuracy = float(accuracy_score(ytrue, preds))
        macro_f1 = float(f1_score(ytrue, preds, average='macro'))
    except Exception:
        auroc = auprc = accuracy = macro_f1 = float('nan')
    return {"auroc": auroc, "auprc": auprc, "accuracy": accuracy, "macro_f1": macro_f1}


def build_graphs(train_npz: str, cfg: dict, save_preprocessor_dir: str | None = None):
    d = np.load(train_npz, allow_pickle=True)
    Xs = list(d['Xs']); ys = list(d['ys']); xys = list(d['xys']); slide_ids = list(d['slide_ids'])
    gene_names = list(d['gene_names'])
    slides = [{"X": X, "y": y, "xy": xy, "slide_id": sid} for X, y, xy, sid in zip(Xs, ys, xys, slide_ids)]

    # n_hvg 解析
    _n_hvg_cfg = cfg.get('n_hvg', 'all')
    n_hvg_val = len(gene_names) if (isinstance(_n_hvg_cfg, str) and _n_hvg_cfg.lower() == 'all') else int(_n_hvg_cfg)

    pp = Preprocessor(n_hvg=n_hvg_val, pca_dim=cfg['pca_dim'], use_pca=cfg.get('use_pca', True))
    pp.fit(slides, gene_names)
    if save_preprocessor_dir is not None:
        pp.save(save_preprocessor_dir)

    graphs = []
    for s in slides:
        Xp = pp.transform(s['X'], gene_names)
        g = assemble_pyg(Xp, s['xy'], s['y'], cfg)
        g.slide_id = str(s['slide_id'])
        graphs.append(g)

    in_dim = graphs[0].x.shape[1]
    return graphs, in_dim


def train_one_run(graphs, cfg: dict, device: torch.device):
    # 简单划分：最后一个切片为验证集（若仅1个切片，则同时作为训练和验证）
    if len(graphs) > 1:
        train_graphs = graphs[:-1]
        val_graphs = graphs[-1:]
    else:
        train_graphs = graphs
        val_graphs = graphs

    model = SpotoncoGATv2Classifier(
        in_dim=train_graphs[0].x.shape[1],
        hidden=cfg['hidden'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        heads=cfg['heads'],
        use_domain_adv_slide=False,
        use_domain_adv_cancer=False,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    train_loader = PyGDataLoader(train_graphs, batch_size=cfg['batch_size_graphs'], shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_graphs, batch_size=1, shuffle=False, num_workers=0)

    best = {"accuracy": -1, "auroc": float('nan'), "auprc": float('nan'), "macro_f1": float('nan'), "state": None, "epoch": -1}
    patience = 0

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch=getattr(batch, 'batch', None))
            logits = out['logits']
            mask = batch.y >= 0
            if mask.sum() > 0:
                loss = bce(logits[mask], batch.y[mask].float())
            else:
                loss = torch.tensor(0.0, device=device)
            loss.backward(); opt.step()

        # 验证
        model.eval()
        val_logits_list, val_y_list = [], []
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                out_v = model(vb.x, vb.edge_index, batch=getattr(vb, 'batch', None))
                val_logits_list.append(out_v['logits'].detach().cpu())
                val_y_list.append(vb.y.detach().cpu())
        val_logits = torch.cat(val_logits_list, dim=0)
        val_y = torch.cat(val_y_list, dim=0)
        metrics = eval_logits(val_logits, val_y)

        if metrics['accuracy'] > best['accuracy']:
            best = {**metrics, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            patience = 0
        else:
            patience += 1
            if patience >= cfg['early_patience']:
                break

    return best


def main():
    p = argparse.ArgumentParser(description='Simple training for GATv2 model')
    p.add_argument('--train_npz', required=True, help='训练数据NPZ，包含 Xs, ys, xys, slide_ids, gene_names')
    p.add_argument('--artifacts_dir', default='artifacts', help='训练产物输出目录')
    # 模型/图相关参数
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--knn_k', type=int, default=6)
    p.add_argument('--gaussian_sigma_factor', type=float, default=1.0)
    p.add_argument('--lap_pe_dim', type=int, default=16)
    p.add_argument('--concat_lap_pe', type=int, choices=[0, 1], default=1)
    p.add_argument('--lap_pe_use_gaussian', type=int, choices=[0, 1], default=0)
    # 预处理
    p.add_argument('--use_pca', type=int, choices=[0, 1], default=1)
    p.add_argument('--pca_dim', type=int, default=64)
    p.add_argument('--n_hvg', default='all', help="高变基因数量，或'all' 使用全部基因")
    # 优化与训练
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--early_patience', type=int, default=30)
    p.add_argument('--batch_size_graphs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    # 设备
    p.add_argument('--device', default=None, help='cuda 或 cpu（默认自动检测）')

    args = p.parse_args()

    # 设备
    device = torch.device(args.device) if isinstance(args.device, str) and args.device in ('cpu', 'cuda') else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    # cfg 汇总（用于记录到 meta.json，推理时也会读取）
    cfg = {
        'hidden': args.hidden,
        'num_layers': args.num_layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'knn_k': args.knn_k,
        'gaussian_sigma_factor': args.gaussian_sigma_factor,
        'lap_pe_dim': args.lap_pe_dim,
        'concat_lap_pe': bool(args.concat_lap_pe),
        'lap_pe_use_gaussian': bool(args.lap_pe_use_gaussian),
        'use_pca': bool(args.use_pca),
        'pca_dim': args.pca_dim,
        'n_hvg': args.n_hvg,
        'epochs': args.epochs,
        'early_patience': args.early_patience,
        'batch_size_graphs': args.batch_size_graphs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
    }

    # 构图 + 保存预处理器
    graphs, in_dim = build_graphs(args.train_npz, cfg, save_preprocessor_dir=args.artifacts_dir)

    # 训练
    best = train_one_run(graphs, cfg, device)

    # 保存最优模型
    os.makedirs(args.artifacts_dir, exist_ok=True)
    model = SpotoncoGATv2Classifier(
        in_dim=in_dim,
        hidden=cfg['hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], heads=cfg['heads'],
        use_domain_adv_slide=False, use_domain_adv_cancer=False,
    ).to(device)
    model.load_state_dict(best['state'])
    save_model(model, args.artifacts_dir)

    # 保存元信息（推理时会读取 cfg）
    meta = {"cfg": cfg, "best_epoch": best['epoch'], "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best.items() if k in ('auroc','auprc','accuracy','macro_f1')}}
    save_json(meta, os.path.join(args.artifacts_dir, 'meta.json'))
    print(f"Saved artifacts to: {args.artifacts_dir}")


if __name__ == '__main__':
    main()