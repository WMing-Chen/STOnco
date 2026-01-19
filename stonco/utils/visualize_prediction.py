#!/usr/bin/env python3
"""
可视化单个切片的预测结果：
- 左图：真实标签（y）空间分布
- 右图：预测标签（probs>=threshold）空间分布
- 图标题处标注准确率（若存在 y）

用法示例：
  python Spotonco/visualize_prediction.py \
    --train_npz Spotonco/synthetic_data/train_data.npz \
    --artifacts_dir Spotonco/artifacts_synth \
    --slide_idx -1 \
    --out_svg Spotonco/synthetic_data/vis_val_slide.svg

或（若单切片 npz 内含 y）：
  python Spotonco/visualize_prediction.py \
    --npz Spotonco/synthetic_data/one_slide_with_y.npz \
    --artifacts_dir Spotonco/artifacts_synth \
    --out_svg Spotonco/synthetic_data/vis_one_slide.svg
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 后端设为无交互，便于服务器保存图片
import matplotlib.pyplot as plt
from torch_geometric.data import Data as PyGData
# 新增：读取验证集坐标与标签
import pandas as pd

from stonco.utils.preprocessing import Preprocessor, GraphBuilder
from stonco.core.models import STOnco_Classifier
from stonco.utils.utils import load_model_state_dict, load_json


def assemble_pyg(Xp, xy, cfg):
    gb = GraphBuilder(knn_k=cfg['knn_k'], gaussian_sigma_factor=cfg['gaussian_sigma_factor'])
    edge_index, edge_weight, _ = gb.build_knn(xy)
    # lapPE（可选：使用高斯权重并控制是否拼接）
    if cfg.get('lap_pe_dim', 0) and cfg.get('lap_pe_dim', 0) > 0:
        pe = gb.lap_pe(edge_index, Xp.shape[0], k=cfg['lap_pe_dim'],
                       edge_weight=edge_weight if cfg.get('lap_pe_use_gaussian', False) else None,
                       use_gaussian_weights=cfg.get('lap_pe_use_gaussian', False))
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


def load_one_slide(train_npz=None, single_npz=None, slide_idx=-1):
    if train_npz:
        data = np.load(train_npz, allow_pickle=True)
        Xs = list(data['Xs'])
        ys = list(data['ys'])
        xys = list(data['xys'])
        slide_ids = list(data['slide_ids'])
        # 选择指定切片
        X = Xs[slide_idx]
        y = ys[slide_idx]
        xy = xys[slide_idx]
        gene_names = list(data['gene_names'])
        sid = slide_ids[slide_idx]
        barcodes = None  # 训练NPZ不包含barcodes
        return X, y, xy, gene_names, sid, barcodes
    else:
        data = np.load(single_npz, allow_pickle=True)
        X = data['X']
        xy = data['xy']
        gene_names = list(data['gene_names'])
        y = data['y'] if 'y' in data.files else None
        # 从NPZ读取sample_id和barcodes
        sid = str(data.get('sample_id', Path(single_npz).stem))
        barcodes = data.get('barcodes', None)
        return X, y, xy, gene_names, sid, barcodes

# 新增：从验证集目录读取真实标签，并按 (row,col) 与 xy 对齐
def load_true_labels_from_valdir(val_root: str, slide_id: str, xy: np.ndarray, x_col: str = 'row', y_col: str = 'col', label_col: str = 'true_label') -> np.ndarray:
    slide_dir = os.path.join(val_root, slide_id)
    # 自动寻找 *coordinates.csv 文件
    coord_csv = None
    if os.path.isdir(slide_dir):
        for f in os.listdir(slide_dir):
            if f.lower().endswith('coordinates.csv'):
                coord_csv = os.path.join(slide_dir, f)
                break
    if coord_csv is None:
        # 退化尝试：{sid}_coordinates.csv
        cand = os.path.join(slide_dir, f"{slide_id}_coordinates.csv")
        if os.path.exists(cand):
            coord_csv = cand
    if coord_csv is None:
        raise FileNotFoundError(f"Coordinates CSV not found for slide '{slide_id}' under {slide_dir}")

    df = pd.read_csv(coord_csv)
    # 列名大小写无关解析
    cmap = {c.lower(): c for c in df.columns}
    def _resolve(name: str) -> str:
        c = cmap.get(name.lower())
        if c is None:
            raise ValueError(f"Column '{name}' not found in {coord_csv}. Available: {list(df.columns)}")
        return c
    x_name = _resolve(x_col)
    y_name = _resolve(y_col)
    lbl_name = _resolve(label_col)
    # 规范为整数坐标与整数标签
    xr = pd.to_numeric(df[x_name], errors='coerce').round().astype('Int64')
    yr = pd.to_numeric(df[y_name], errors='coerce').round().astype('Int64')
    yl = pd.to_numeric(df[lbl_name], errors='coerce').fillna(0).astype(int)
    # 建立 (row,col)->label 字典
    key_series = list(zip(xr.astype('Int64'), yr.astype('Int64')))
    coord2y = { (int(r), int(c)): int(lbl) for (r,c), lbl in zip(key_series, yl) if pd.notna(r) and pd.notna(c) }
    # 将 npz 的 xy（float）四舍五入到整数并查找标签
    xy_int = np.rint(xy).astype(int)
    ys = [ coord2y.get((int(x), int(y)), -1) for x, y in xy_int ]
    return np.asarray(ys, dtype=int)


def main():
    parser = argparse.ArgumentParser(description='空间分布可视化（真实 vs 预测）')
    parser.add_argument('--train_npz', default=None, help='训练 npz（包含多个切片）')
    parser.add_argument('--npz', default=None, help='单切片 npz（可选，若包含 y 则可计算准确率）')
    parser.add_argument('--npz_glob', default=None, help='批量处理：NPZ文件glob模式，如 "val_npz/*.npz"')
    parser.add_argument('--slide_idx', type=int, default=-1, help='当使用 train_npz 时选择的切片索引，默认 -1（最后一张）')
    parser.add_argument('--artifacts_dir', default='Spotonco/artifacts_synth', help='模型与预处理产物目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    parser.add_argument('--out_svg', default='Spotonco/synthetic_data/vis_slide.svg', help='输出 SVG 路径')
    parser.add_argument('--out_dir', default=None, help='批量处理时的输出目录')
    # 新增：验证集真实标签目录与列名
    parser.add_argument('--val_root', default='Spotonco/data/ST_validation_datasets', help='验证集根目录，子目录为每个切片（含 *coordinates.csv，含 true_label 列）')
    parser.add_argument('--xy_cols', nargs=2, default=['row', 'col'], metavar=('X_COL', 'Y_COL'), help='coordinates.csv 中表示坐标的列名（默认 row col）')
    parser.add_argument('--label_col', default='true_label', help='coordinates.csv 中标签列名（默认 true_label）')
    args = parser.parse_args()

    # 检查参数组合
    mode_count = sum([args.train_npz is not None, args.npz is not None, args.npz_glob is not None])
    if mode_count != 1:
        raise ValueError('请在 --train_npz、--npz、--npz_glob 中选择一个')
    
    # 批量处理模式
    if args.npz_glob is not None:
        import glob
        npz_files = sorted(glob.glob(args.npz_glob))
        if not npz_files:
            raise FileNotFoundError(f'No NPZ files found matching: {args.npz_glob}')
        if args.out_dir is None:
            raise ValueError('批量处理模式需要指定 --out_dir')
        
        # 批量处理每个NPZ文件
        for npz_file in npz_files:
            process_single_slide(npz_file, args)
        return

    # 单个处理模式
    process_single_slide_main(args)


def process_single_slide(npz_file, args):
    """处理单个NPZ文件并生成SVG"""
    # 加载数据
    X, y, xy, gene_names, sid, barcodes = load_one_slide(single_npz=npz_file)
    
    # 从meta.json加载配置，保持与训练一致
    meta = load_json(os.path.join(args.artifacts_dir, 'meta.json'))
    cfg = dict(meta.get('cfg', {}))
    
    # 输出路径
    out_svg = os.path.join(args.out_dir, f'{sid}.svg')
    
    # 处理可视化
    visualize_slide(X, xy, gene_names, sid, y, cfg, args, out_svg)
    print(f'Saved visualization for {sid} to {out_svg}')


def process_single_slide_main(args):
    """处理单个切片的主函数"""
    # 加载数据（一张切片）
    X, y, xy, gene_names, sid, barcodes = load_one_slide(train_npz=args.train_npz, single_npz=args.npz, slide_idx=args.slide_idx)
    
    # 从meta.json加载配置，保持与训练一致
    meta = load_json(os.path.join(args.artifacts_dir, 'meta.json'))
    cfg = dict(meta.get('cfg', {}))
    
    # 处理可视化
    visualize_slide(X, xy, gene_names, sid, y, cfg, args, args.out_svg)


def visualize_slide(X, xy, gene_names, sid, y, cfg, args, out_svg):
    """可视化单个切片的核心逻辑"""
    # 兼容旧模型：若缺失新字段则给默认
    cfg.setdefault('lap_pe_dim', 16)
    cfg.setdefault('edge_attr_dim', 0)
    cfg.setdefault('use_edge_attr', False)

    # 若单切片 npz 无 y，尝试从验证集目录读取 true_label
    if y is None and args.val_root is not None:
        try:
            x_col, y_col = args.xy_cols
            y = load_true_labels_from_valdir(args.val_root, sid, xy, x_col=x_col, y_col=y_col, label_col=args.label_col)
            print(f"Loaded ground-truth labels for slide {sid} from {args.val_root}")
        except Exception as e:
            print(f"Warning: failed to load ground-truth from val_root for slide {sid}: {e}")
            y = None

    # 预处理与图构建
    pp = Preprocessor.load(args.artifacts_dir)
    Xp = pp.transform(X, gene_names)
    data_g = assemble_pyg(Xp, xy, cfg)

    # 构建与加载模型（统一使用 SpotoncoGNNClassifier）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = data_g.x.shape[1]
    model = STOnco_Classifier(
        in_dim=in_dim,
        hidden=cfg['hidden'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        model=cfg['model'],
        heads=cfg.get('heads', 4),
        use_domain_adv=False,
    )
    _ = model.load_state_dict(load_model_state_dict(args.artifacts_dir, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    # 预测概率
    with torch.no_grad():
        g = data_g.to(device)
        out = model(g.x, g.edge_index, batch=getattr(g, 'batch', None), edge_weight=getattr(g, 'edge_weight', None))
        logits = out['logits']
        probs = torch.sigmoid(logits).cpu().numpy()
    pred = (probs >= args.threshold).astype(int)

    # 计算准确率（若有 y）
    acc = None
    if y is not None:
        y_arr = np.asarray(y)
        mask = y_arr >= 0
        if mask.sum() > 0:
            acc = (pred[mask] == y_arr[mask]).mean()

    # 作图
    cmap = {0: '#4472C4', 1: '#D9534F'}  # 非肿瘤: 蓝，肿瘤: 红
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # 左：真实标签
    if y is not None:
        y_plot = np.asarray(y)
        colors_true = [cmap[int(v)] if v in (0, 1) else '#BDBDBD' for v in y_plot]
        axes[0].scatter(xy[:, 0], xy[:, 1], c=colors_true, s=6, linewidths=0, alpha=0.9)
        axes[0].set_title('Ground Truth')
    else:
        axes[0].text(0.5, 0.5, 'No ground-truth labels', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Ground Truth (N/A)')

    # 右：预测标签
    colors_pred = [cmap[int(v)] for v in pred]
    axes[1].scatter(xy[:, 0], xy[:, 1], c=colors_pred, s=6, linewidths=0, alpha=0.9)
    axes[1].set_title('Prediction (thr=%.2f)' % args.threshold)

    # 公共外观
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 与显微镜图像坐标一致（可选）

    # 标注准确率
    if acc is not None:
        fig.suptitle(f'Slide: {sid}  |  Accuracy: {acc*100:.2f}%  (n={len(xy)})', fontsize=12)
    else:
        fig.suptitle(f'Slide: {sid}  |  Accuracy: N/A  (no ground-truth)', fontsize=12)

    # 保存 SVG
    out_dir = os.path.dirname(out_svg) or '.'
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_svg, format='svg')
    plt.close(fig)
    print('Saved visualization to', out_svg)


if __name__ == '__main__':
    main()