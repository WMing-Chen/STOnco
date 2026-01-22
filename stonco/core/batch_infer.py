import argparse
import os
import glob
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from stonco.utils.preprocessing import Preprocessor, GraphBuilder
from stonco.utils.utils import load_json
from .models import STOnco_Classifier
from torch_geometric.data import Data as PyGData, DataLoader as PyGDataLoader
from stonco.utils.plot_accuracy_bars import plot_accuracy_bars
# 新增：导入推理引擎
from stonco.core.infer import InferenceEngine


def assemble_pyg(Xp: np.ndarray, xy: np.ndarray, cfg: dict) -> PyGData:
    gb = GraphBuilder(knn_k=cfg['knn_k'], gaussian_sigma_factor=cfg['gaussian_sigma_factor'])
    edge_index, edge_weight, _ = gb.build_knn(xy)
    # lapPE with optional gaussian weighting and concat flag
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

# 新增：基于文件列表的 Dataset，将 NPZ 读取、预处理与图构建放入 __getitem__，以便 DataLoader 多进程并行
class SlideNPZDataset:
    def __init__(self, files, preprocessor: Preprocessor, cfg: dict):
        self.files = list(files)
        self.pp = preprocessor
        self.cfg = cfg
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        d = np.load(f, allow_pickle=True)
        X, xy, gene_names = d['X'], d['xy'], list(d['gene_names'])
        # 读取 barcodes 和 sample_id
        barcodes = d.get('barcodes', None)
        sample_id = str(d.get('sample_id', 'unknown'))
        # 读取 y_true（如果存在）
        y_true = d.get('y', None)
        
        Xp = self.pp.transform(X, gene_names)
        g = assemble_pyg(Xp, xy, self.cfg)
        g.file_name = os.path.basename(f)
        # 保存坐标、barcodes、sample_id 和 y_true 用于回填输出
        g.xy = torch.from_numpy(xy).float()
        g.barcodes = barcodes
        g.sample_id = sample_id
        g.y_true = y_true
        return g

# 通过 DataLoader 进行批处理推理，不再使用单样本函数


def main():
    p = argparse.ArgumentParser(description='Batch inference for multiple NPZ slides')
    p.add_argument('--npz_glob', default=None, help='Glob pattern to NPZ files, e.g. Spotonco/synthetic_data/infer_slide_*.npz')
    p.add_argument('--train_npz', default=None, help='训练NPZ路径（用于内部验证集预测，需配合meta.json的val_ids）')
    p.add_argument('--external_val_dir', default=None, help='外部验证NPZ目录（单切片NPZ）')
    p.add_argument('--artifacts_dir', default='artifacts', help='Directory containing model.pt and preprocessor artifacts')
    p.add_argument('--out_csv', default='batch_preds.csv', help='Output CSV path for concatenated predictions')
    p.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification (default: 0.5)')
    # 新增：CPU线程和数据加载控制参数
    p.add_argument('--num_threads', type=int, default=None, help='设置PyTorch计算线程数（CPU模式下限制核心占用）')
    p.add_argument('--num_workers', type=int, default=0, help='DataLoader数据加载工作进程数（默认0为单进程）')
    # 新增：绘图控制参数
    p.add_argument('--no_plot', action='store_true', help='禁用自动绘制准确率柱状图（默认开启绘图）')
    p.add_argument('--plot_out', type=str, default=None, help='绘图输出PNG路径（默认与out_csv同目录，文件名为{model_name}_accuracy_bars.png）')
    p.add_argument('--model_name', type=str, default=None, help='图表标题的模型名（默认从meta.json的cfg.model推断）')
    # 新增：解释性输出（默认开启，可用 --no_explain 关闭）
    p.add_argument('--explain_saliency', action='store_true', default=True, help='为每张切片计算并保存基因重要性向量（默认开启）')
    p.add_argument('--no_explain', action='store_false', dest='explain_saliency', help='关闭解释性输出')
    p.add_argument('--explain_method', choices=['ig', 'saliency'], default='ig', help='选择解释方法')
    p.add_argument('--ig_steps', type=int, default=50, help='IG步数，默认50')
    p.add_argument('--gene_attr_out_dir', type=str, default=None, help='基因重要性CSV输出目录（默认：与out_csv同目录的gene_attr子目录）')
    args = p.parse_args()

    # 应用CPU线程设置
    if args.num_threads is not None and args.num_threads > 0:
        try:
            torch.set_num_threads(args.num_threads)
            torch.set_num_interop_threads(max(1, min(args.num_threads, 2)))
            print(f"Set torch threads: intra-op={args.num_threads}, inter-op={max(1, min(args.num_threads, 2))}")
        except Exception as e:
            print(f"Warning: failed to set torch threads: {e}")

    # 使用推理引擎以复用cfg/pp/model
    engine = InferenceEngine(args.artifacts_dir, num_threads=args.num_threads)
    cfg = engine.cfg

    pp = engine.pp
    rows = []
    sample_source = {}

    def _append_rows_from_graph(sample_id, xy_np, probs, y_true, barcodes, source):
        sample_source[sample_id] = source
        for i, p in enumerate(probs):
            barcode = barcodes[i] if barcodes is not None and i < len(barcodes) else f'spot_{i}'
            y_true_val = int(y_true[i]) if y_true is not None and i < len(y_true) else None
            pred_label = 1 if p >= args.threshold else 0
            rows.append({
                'sample_id': sample_id,
                'Barcode': barcode,
                'spot_idx': i,
                'x': float(xy_np[i, 0]) if xy_np is not None else float('nan'),
                'y': float(xy_np[i, 1]) if xy_np is not None else float('nan'),
                'p_tumor': float(p),
                'pred_label': pred_label,
                'y_true': y_true_val,
                'threshold': args.threshold,
            })

    def _predict_external_files(files, source_label):
        if not files:
            return
        dataset = SlideNPZDataset(files, pp, cfg)
        loader = PyGDataLoader(dataset, batch_size=1, shuffle=False, num_workers=max(0, args.num_workers))
        for batch in loader:
            probs = engine.predict_proba(batch)

            file_attr = getattr(batch, 'file_name', None)
            if isinstance(file_attr, (list, tuple)):
                file_name = file_attr[0]
            elif isinstance(file_attr, str):
                file_name = file_attr
            else:
                file_name = 'unknown'

            xy_attr = getattr(batch, 'xy', None)
            if isinstance(xy_attr, (list, tuple)):
                xy_np = xy_attr[0].cpu().numpy()
            elif isinstance(xy_attr, torch.Tensor):
                xy_np = xy_attr.cpu().numpy()
            else:
                xy_np = None

            barcodes_attr = getattr(batch, 'barcodes', None)
            if isinstance(barcodes_attr, (list, tuple)):
                barcodes = barcodes_attr[0]
            else:
                barcodes = barcodes_attr

            sample_id_attr = getattr(batch, 'sample_id', None)
            if isinstance(sample_id_attr, (list, tuple)):
                sample_id = sample_id_attr[0]
            elif isinstance(sample_id_attr, str):
                sample_id = sample_id_attr
            else:
                sample_id = 'unknown'

            y_true_attr = getattr(batch, 'y_true', None)
            if isinstance(y_true_attr, (list, tuple)):
                y_true = y_true_attr[0]
            else:
                y_true = y_true_attr

            if args.explain_saliency:
                try:
                    attr_gene = engine.compute_gene_attr_for_graph(batch, method=args.explain_method, ig_steps=args.ig_steps)
                    out_dir = args.gene_attr_out_dir or os.path.join(os.path.dirname(args.out_csv) or '.', 'gene_attr')
                    os.makedirs(out_dir, exist_ok=True)
                    tag = sample_id if sample_id and sample_id != 'unknown' else os.path.splitext(file_name)[0]
                    out_npz = os.path.join(out_dir, f'per_slide_gene_saliency_{tag}.npz')
                    np.savez(out_npz, gene_names=np.array(engine.pp.hvg), attr_gene=attr_gene)
                    out_csv = os.path.join(out_dir, f'per_slide_gene_saliency_{tag}.csv')
                    pd.DataFrame({'gene': np.array(engine.pp.hvg), 'attr': attr_gene}).to_csv(out_csv, index=False)
                    print(f'Saved gene importance CSV to {out_csv}')
                except Exception as e:
                    print(f'[Explain] Failed to compute/save gene vector for {file_name}: {e}')

            _append_rows_from_graph(sample_id, xy_np, probs, y_true, barcodes, source_label)

    def _predict_internal_from_train_npz(train_npz_path):
        if not train_npz_path:
            return
        meta = load_json(os.path.join(args.artifacts_dir, 'meta.json'))
        val_ids = set(map(str, meta.get('val_ids', [])))
        if not val_ids:
            print('[Warn] meta.json has no val_ids; skip internal validation prediction.')
            return
        d = np.load(train_npz_path, allow_pickle=True)
        slide_ids = list(map(str, list(d['slide_ids'])))
        Xs = list(d['Xs']); ys = list(d['ys']); xys = list(d['xys'])
        gene_names = list(d['gene_names'])
        id_to_idx = {sid: i for i, sid in enumerate(slide_ids)}
        missing = [sid for sid in val_ids if sid not in id_to_idx]
        if missing:
            print(f'[Warn] {len(missing)} val_ids not found in train_npz: {missing[:5]}')
        for sid in slide_ids:
            if sid not in val_ids:
                continue
            idx = id_to_idx[sid]
            X = Xs[idx]
            xy = xys[idx]
            y_true = ys[idx] if ys is not None else None
            Xp = engine.pp.transform(X, gene_names)
            g = assemble_pyg(Xp, xy, cfg)
            probs = engine.predict_proba(g)
            _append_rows_from_graph(sid, xy, probs, y_true, None, 'internal')

    external_files = []
    if args.npz_glob:
        external_files.extend(sorted(glob.glob(args.npz_glob)))
    if args.external_val_dir:
        external_files.extend(sorted(glob.glob(os.path.join(args.external_val_dir, '*.npz'))))
    external_files = sorted(set(external_files))

    if not external_files and not args.train_npz:
        raise FileNotFoundError('No input provided. Use --npz_glob and/or --external_val_dir and/or --train_npz.')

    _predict_internal_from_train_npz(args.train_npz)
    _predict_external_files(external_files, 'external')

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(args.out_csv, index=False)
    print('Saved batch predictions to', args.out_csv)

    # 样本级汇总
    try:
        summary_rows = []
        for sample_id, sub in pred_df.groupby('sample_id'):
            probs = sub['p_tumor'].astype(float).values
            pred_labels = sub['pred_label'].astype(int).values
            y_mask = sub['y_true'].notna().values
            y_true_arr = sub.loc[y_mask, 'y_true'].astype(int).values
            pred_for_y = pred_labels[y_mask]
            probs_for_y = probs[y_mask]
            row = {
                'sample_id': sample_id,
                'source': sample_source.get(sample_id, 'external'),
                'n_spots': int(len(sub)),
                'threshold': float(sub['threshold'].iloc[0]) if 'threshold' in sub else float('nan'),
                'accuracy': float('nan'),
                'auroc': float('nan'),
                'auprc': float('nan'),
                'macro_f1': float('nan'),
            }
            if len(y_true_arr) > 0:
                try:
                    row['accuracy'] = float(accuracy_score(y_true_arr, pred_for_y))
                except Exception:
                    row['accuracy'] = float('nan')
                try:
                    row['auroc'] = float(roc_auc_score(y_true_arr, probs_for_y))
                except Exception:
                    row['auroc'] = float('nan')
                try:
                    row['auprc'] = float(average_precision_score(y_true_arr, probs_for_y))
                except Exception:
                    row['auprc'] = float('nan')
                try:
                    row['macro_f1'] = float(f1_score(y_true_arr, pred_for_y, average='macro'))
                except Exception:
                    row['macro_f1'] = float('nan')
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(os.path.dirname(args.out_csv) or '.', 'batch_preds_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print('Saved batch prediction summary to', summary_path)
    except Exception as e:
        print(f'Warning: failed to write summary CSV: {e}')

    # 自动绘制准确率柱状图（除非 --no_plot）
    if not args.no_plot:
        try:
            pred_df = pd.read_csv(args.out_csv)
            if 'sample_id' not in pred_df.columns:
                print(f"Skip plotting: 'sample_id' column not found in {args.out_csv}")
            else:
                slide_ids = sorted(pred_df['sample_id'].dropna().unique().tolist())
                if len(slide_ids) == 0:
                    print("Skip plotting: no slides found in predictions")
                else:
                    model_name = args.model_name if args.model_name else cfg.get('model', 'model')
                    out_dir = os.path.dirname(args.out_csv) or '.'
                    plot_out = args.plot_out if args.plot_out else os.path.join(out_dir, f"{model_name}_accuracy_bars.png")
                    mean_acc, total_spots = plot_accuracy_bars(
                        model_name=model_name,
                        pred_csv=args.out_csv,
                        slide_ids=slide_ids,
                        out_path=plot_out,
                        threshold=args.threshold
                    )
                    print(f"Saved accuracy bar plot to: {plot_out}")
                    print(f"Mean accuracy: {mean_acc*100:.2f}%, Total spots: {total_spots}")
        except Exception as e:
            print(f"Warning: plotting failed: {e}")


if __name__ == '__main__':
    main()
