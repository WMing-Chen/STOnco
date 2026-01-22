import argparse, os, numpy as np, torch
from stonco.utils.preprocessing import Preprocessor, GraphBuilder
from .models import STOnco_Classifier, grad_reverse
from stonco.utils.utils import save_model, save_json
from torch_geometric.data import Data as PyGData, DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path
import time
import random

# HPO相关功能已迁移至 train_hpo.py（此处不再导入optuna）

# -------------------------- 新增：基于癌种的划分辅助函数 --------------------------
def _load_cancer_labels():
    """加载 data/cancer_sample_labels.csv，返回癌种与批次标签映射。
    返回：id2type, type2ids, id2batch, batch2ids
    若CSV缺失，将返回空映射，并在后续基于前缀进行兜底推断并打印提示信息。
    """
    csv_path = Path(__file__).resolve().parents[2] / 'data' / 'cancer_sample_labels.csv'
    if not csv_path.exists():
        print(f"[Warn] Cancer label CSV not found: {csv_path}. Will fallback to slide_id prefix or UNKNOWN.")
        return {}, {}
    df = pd.read_csv(csv_path)
    id2type = {str(r['sample_id']): str(r['cancer_type']) for _, r in df.iterrows()}
    id2batch = {}
    if 'Batch_id' in df.columns:
        id2batch = {str(r['sample_id']): str(r['Batch_id']) for _, r in df.iterrows()}
    else:
        print(f"[Warn] Column 'Batch_id' not found in {csv_path}. Will fallback to slide_id for batch domain.")
    type2ids = {}
    for sid, ctype in id2type.items():
        type2ids.setdefault(ctype, []).append(sid)
    batch2ids = {}
    for sid, bid in id2batch.items():
        batch2ids.setdefault(bid, []).append(sid)
    return id2type, type2ids, id2batch, batch2ids

def _build_type_to_present_ids(present_ids):
    """基于当前NPZ中的 slide_ids 构建 {cancer_type:[present_ids]} 和 {present_id:cancer_type}."""
    id2type_all, _, _, _ = _load_cancer_labels()
    type2present = {}
    present_id2type = {}
    fallback_cache = set()
    for sid in present_ids:
        s = str(sid)
        ctype = _resolve_cancer_type(s, id2type_all, fallback_cache)
        present_id2type[s] = ctype
        type2present.setdefault(ctype, []).append(s)
    return present_id2type, type2present


def _resolve_batch_id(sample_id: str, id2batch: dict, fallback_cache: set) -> str:
    """解析 batch_id；若缺失则回退到 slide_id，并提示一次。"""
    bid = id2batch.get(sample_id)
    if bid is None or str(bid).strip() == '' or str(bid).lower() == 'nan':
        if sample_id not in fallback_cache:
            print(f"[Warn] Batch_id missing for sample_id '{sample_id}'. Fallback to slide_id as batch domain.")
            fallback_cache.add(sample_id)
        return sample_id
    return str(bid)


def _resolve_cancer_type(sample_id: str, id2type: dict, fallback_cache: set) -> str:
    """解析 cancer_type；若缺失则回退到前缀或 UNKNOWN，并提示一次。"""
    ctype = id2type.get(sample_id, None)
    if ctype is None:
        prefix = ''.join([ch for ch in sample_id if ch.isalpha()])
        if prefix:
            ctype = prefix
            if sample_id not in fallback_cache:
                print(f"[Info] sample_id '{sample_id}' not found in CSV, fallback to prefix '{prefix}'.")
                fallback_cache.add(sample_id)
        else:
            ctype = 'UNKNOWN'
            if sample_id not in fallback_cache:
                print(f"[Info] sample_id '{sample_id}' not found in CSV and no alpha prefix; fallback to 'UNKNOWN'.")
                fallback_cache.add(sample_id)
    return ctype

def _stratified_single_split(present_ids, rng, val_ratio=0.2):
    """按比例划分验证集（保底每癌种1张，n=1则只进训练）。
    返回 (train_ids, val_ids)
    """
    present_id2type, type2present = _build_type_to_present_ids(present_ids)
    val_ids = []
    for ctype in sorted(type2present.keys()):
        ids = list(type2present[ctype])
        n = len(ids)
        if n <= 1:
            continue
        raw = int(round(n * float(val_ratio)))
        n_val = max(1, min(n - 1, raw))
        if n_val == 1:
            val_ids.append(rng.choice(ids))
        else:
            val_ids.extend(rng.sample(ids, n_val))
    train_ids = [s for s in map(str, present_ids) if s not in set(val_ids)]
    return train_ids, val_ids, present_id2type

def _k_random_combinations(present_ids, k, rng, val_ratio=0.2):
    """生成k个不同的组合：按比例为每个癌种随机选取若干样本作为验证集。
    返回 folds 列表，每个元素为 (train_ids, val_ids)
    """
    present_id2type, type2present = _build_type_to_present_ids(present_ids)
    per_type_nval = {}
    for ctype, ids in type2present.items():
        n = len(ids)
        if n <= 1:
            per_type_nval[ctype] = 0
            continue
        raw = int(round(n * float(val_ratio)))
        per_type_nval[ctype] = max(1, min(n - 1, raw))

    folds = []
    seen = set()
    max_attempts = k * 50
    attempts = 0
    types_sorted = sorted(type2present.keys())
    while len(folds) < k and attempts < max_attempts:
        attempts += 1
        val_ids = []
        for ctype in types_sorted:
            ids = list(type2present[ctype])
            n_val = per_type_nval.get(ctype, 0)
            if n_val <= 0:
                continue
            if n_val == 1:
                val_ids.append(rng.choice(ids))
            else:
                val_ids.extend(rng.sample(ids, n_val))
        key = tuple(sorted(val_ids))
        if key in seen:
            continue
        seen.add(key)
        train_ids = [s for s in map(str, present_ids) if s not in set(val_ids)]
        folds.append((train_ids, val_ids))
    return folds, present_id2type

# ---------------------------------------------------------------------------

def assemble_pyg(Xp, xy, y, cfg):
    from stonco.utils.preprocessing import GraphBuilder
    gb = GraphBuilder(knn_k=cfg['knn_k'], gaussian_sigma_factor=cfg['gaussian_sigma_factor'])
    edge_index, edge_weight, mean_nd = gb.build_knn(xy)
    # 计算PE（可选使用高斯距离权重）
    if cfg.get('lap_pe_dim', 0) and cfg.get('lap_pe_dim', 0) > 0:
        pe = gb.lap_pe(edge_index, Xp.shape[0], k=cfg['lap_pe_dim'],
                       edge_weight=edge_weight if cfg.get('lap_pe_use_gaussian', False) else None,
                       use_gaussian_weights=cfg.get('lap_pe_use_gaussian', False))
    else:
        pe = None
    # 控制是否拼接PE
    if cfg.get('concat_lap_pe', True) and pe is not None:
        x = np.hstack([Xp, pe]).astype('float32')
    else:
        x = Xp.astype('float32')
    data = PyGData(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index).long(), edge_weight=torch.from_numpy(edge_weight).float(), y=torch.from_numpy(y).long())
    data.num_nodes = x.shape[0]
    return data

def eval_logits(logits, y, return_predictions=False):
    """计算验证指标，支持binary classification
    Args:
        logits: 模型输出logits
        y: 真实标签
        return_predictions: 是否返回预测结果
    Returns:
        dict with auroc, auprc, accuracy, macro_f1
    """
    mask = y>=0
    if mask.sum()==0:
        result = {'auroc': float('nan'), 'auprc': float('nan'), 'accuracy': float('nan'), 'macro_f1': float('nan')}
        if return_predictions:
            result['predictions'] = None
        return result
    
    ytrue = y[mask].cpu().numpy()
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    
    try:
        auroc = float(roc_auc_score(ytrue, probs))
        auprc = float(average_precision_score(ytrue, probs))
        accuracy = float(accuracy_score(ytrue, preds))
        macro_f1 = float(f1_score(ytrue, preds, average='macro'))
    except Exception:
        auroc = auprc = accuracy = macro_f1 = float('nan')
    
    result = {'auroc': auroc, 'auprc': auprc, 'accuracy': accuracy, 'macro_f1': macro_f1}
    if return_predictions:
        result['predictions'] = preds
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', required=True)
    parser.add_argument('--artifacts_dir', default='artifacts')
    
    # 现有可选参数
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--early_patience', type=int, default=None, help='早停耐心值，<=0 表示关闭早停')
    parser.add_argument('--batch_size_graphs', type=int, default=None)
    parser.add_argument('--model', choices=['gatv2', 'sage', 'gcn'], default=None, help='选择GNN主干，默认gatv2')
    parser.add_argument('--heads', type=int, default=None, help='GATv2的多头数（仅对gatv2有效）')
    parser.add_argument('--concat_lap_pe', type=int, choices=[0,1], default=None, help='是否将lapPE拼接至节点特征（1/0）')
    parser.add_argument('--lap_pe_use_gaussian', type=int, choices=[0,1], default=None, help='lapPE是否使用高斯边权（1/0）')
    parser.add_argument('--lap_pe_dim', type=int, default=None, help='lapPE维度（>0表示启用）')
    parser.add_argument('--num_threads', type=int, default=None, help='设置PyTorch计算线程数（CPU模式下限制核心占用）')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader数据加载工作进程数')
    parser.add_argument('--use_pca', type=int, choices=[0, 1], default=None, help='是否使用PCA（1/0）')
    # 新增：HVG数量控制（支持数值或'all'；默认'all'表示使用全部基因）
    parser.add_argument('--n_hvg', default='all', help="高变基因数量，或'all'使用全部基因（默认'all'）")
    # 新增：从JSON加载配置（支持meta风格或扁平字典）
    parser.add_argument('--config_json', default=None, help='从JSON文件加载超参数配置（支持 {"cfg": {...}} 或直接扁平字典）')
    
    # 新增：网络结构与优化器超参数
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--hidden', type=int, default=None, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=None, help='GNN层数')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout比例')
    
    # 新增：设备控制
    parser.add_argument('--device', default=None, help='指定设备（cpu/cuda）,不指定则自动检测')
    
    # 新增：双域对抗控制与权重（新参数优先级最高）
    parser.add_argument('--use_domain_adv_slide', type=int, choices=[0,1], default=None, help='启用/关闭批次域对抗（1/0）')
    parser.add_argument('--use_domain_adv_cancer', type=int, choices=[0,1], default=None, help='启用/关闭癌种域对抗（1/0）')
    parser.add_argument('--lambda_slide', type=float, default=None, help='批次域对抗损失权重')
    parser.add_argument('--lambda_cancer', type=float, default=None, help='癌种域对抗损失权重')
    
    # 新增：癌种分层与K折/LOCO
    parser.add_argument('--stratify_by_cancer', action='store_true', default=True, help='启用癌种分层划分：按比例分配验证集且每癌种保底1张（n=1除外）')
    parser.add_argument('--no_stratify_by_cancer', action='store_false', dest='stratify_by_cancer', help='关闭癌种分层，使用简单划分（最后1张为验证）')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例（默认0.2，按癌种分配且保底1张）')
    parser.add_argument('--kfold_cancer', type=int, default=None, help='整数输入，基于癌种的K折（按比例划分验证集并随机组合）。产物统一保存至artifacts_dir同级目录下，并在/kfold_val/kfold_summary.csv 汇总指标；这里的artifacts_dir 仅决定保存根目录。')
    parser.add_argument('--split_seed', type=int, default=42, help='分层/交叉验证随机种子')
    parser.add_argument('--split_test_only', action='store_true', help='仅测试划分逻辑，不进行训练，打印每折的统计信息')
    parser.add_argument('--leave_one_cancer_out', action='store_true', help='启用留一癌种评估模式（对每个癌种单独训练：其为验证，其余为训练）')
    parser.add_argument('--val_sample_dir', default=None, help='外部验证样本目录（单切片NPZ），将与内部验证集共同评估')

    # 新增：解释性输出（基因重要性）
    parser.add_argument('--explain_saliency', action='store_true', default=True, help='训练结束后基于最终模型计算并保存基因重要性（总体，不分癌种）（默认开启）')
    parser.add_argument('--no_explain', action='store_false', dest='explain_saliency', help='关闭解释性输出')
    parser.add_argument('--explain_method', choices=['ig', 'saliency'], default='ig', help='选择解释方法：集成梯度(ig)或显著性图(saliency)')
    parser.add_argument('--ig_steps', type=int, default=50, help='IG步数，默认50')

    # 新增：保存Loss组件
    parser.add_argument('--save_loss_components', type=int, choices=[0,1], default=1,
                        help='保存Loss组件到CSV (0/1, 默认: 1)')
    # 新增：训练曲线保存开关
    parser.add_argument('--save_train_curves', type=int, choices=[0,1], default=1,
                        help='保存训练曲线SVG (0/1, 默认: 1)')

    args = parser.parse_args()

    cfg = {'pca_dim':64, 'lap_pe_dim':16, 'knn_k':6, 'gaussian_sigma_factor':1.0, 'hidden':128, 'num_layers':3, 'dropout':0.3, 'model':'gatv2', 'heads':4, 'use_domain_adv':True, 'domain_lambda':0.3, 'lr':1e-3, 'weight_decay':1e-4, 'epochs':100, 'batch_size_graphs':2, 'early_patience':30,
           # 控制项
           'use_pca': False,
           'concat_lap_pe': True,
           'lap_pe_use_gaussian': False,
           # 新增：双域默认配置（新字段）。为体现优先级，新字段默认设为None或False，稍后进行兼容性映射
           'use_domain_adv_slide': None,   # 若None，将回退到旧字段 use_domain_adv
           'use_domain_adv_cancer': True, # 默认开启
           'lambda_slide': None,           # 若None，将回退到 domain_lambda
           'lambda_cancer': None,           # 若None，将回退到 domain_lambda
           # 新增：HVG控制
           'n_hvg': 'all'
           }

    # 先从JSON加载作为基准配置（若提供）
    if args.config_json is not None:
        try:
            with open(args.config_json, 'r', encoding='utf-8') as f:
                cfg_json = json.load(f)
            if isinstance(cfg_json, dict):
                if 'cfg' in cfg_json and isinstance(cfg_json['cfg'], dict):
                    cfg.update(cfg_json['cfg'])
                else:
                    cfg.update(cfg_json)
                print(f"Loaded config from {args.config_json}")
            else:
                print(f"Warning: config_json is not a dict: {args.config_json}")
        except Exception as e:
            print(f"Warning: failed to load config_json {args.config_json}: {e}")
 
    # 覆盖配置以支持快速实验和HPO
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.early_patience is not None:
        cfg['early_patience'] = args.early_patience
    if args.batch_size_graphs is not None:
        cfg['batch_size_graphs'] = args.batch_size_graphs
    if args.model is not None:
        cfg['model'] = args.model
    if args.heads is not None:
        cfg['heads'] = args.heads
    if args.concat_lap_pe is not None:
        cfg['concat_lap_pe'] = bool(args.concat_lap_pe)
    if args.lap_pe_use_gaussian is not None:
        cfg['lap_pe_use_gaussian'] = bool(args.lap_pe_use_gaussian)
    if args.lap_pe_dim is not None:
        cfg['lap_pe_dim'] = args.lap_pe_dim
    if args.use_pca is not None:
        cfg['use_pca'] = bool(args.use_pca)
    # 新增：覆盖 HVG 数量（字符串'all'或可解析为整数的字符串/整数）
    if getattr(args, 'n_hvg', None) is not None:
        cfg['n_hvg'] = args.n_hvg
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.weight_decay is not None:
        cfg['weight_decay'] = args.weight_decay
    if args.hidden is not None:
        cfg['hidden'] = args.hidden
    if args.num_layers is not None:
        cfg['num_layers'] = args.num_layers
    if args.dropout is not None:
        cfg['dropout'] = args.dropout

    # 新增：双域对抗参数（新命令行优先级最高）
    if args.use_domain_adv_slide is not None:
        cfg['use_domain_adv_slide'] = bool(args.use_domain_adv_slide)
    if args.use_domain_adv_cancer is not None:
        cfg['use_domain_adv_cancer'] = bool(args.use_domain_adv_cancer)
    if args.lambda_slide is not None:
        cfg['lambda_slide'] = float(args.lambda_slide)
    if args.lambda_cancer is not None:
        cfg['lambda_cancer'] = float(args.lambda_cancer)

    # 兼容性映射与默认值填充（优先级：新CLI > 旧CLI/旧cfg字段 > 新cfg字段 > 默认）
    # use_domain_adv_slide: 若未显式指定，回退到旧字段 use_domain_adv
    if cfg.get('use_domain_adv_slide', None) is None:
        cfg['use_domain_adv_slide'] = bool(cfg.get('use_domain_adv', False))
    # use_domain_adv_cancer: 若未显式指定，保持其在cfg中的值，默认True
    cfg['use_domain_adv_cancer'] = bool(cfg.get('use_domain_adv_cancer', True))
    # lambda_*: 若未显式指定，回退到旧字段 domain_lambda（若不存在则0.3）
    if cfg.get('lambda_slide', None) is None:
        cfg['lambda_slide'] = float(cfg.get('domain_lambda', 0.3))
    if cfg.get('lambda_cancer', None) is None:
        cfg['lambda_cancer'] = float(cfg.get('domain_lambda', 0.3))

    # 设备控制（优先使用命令行指定，否则自动检测；HPO 模式不再强制 CPU）
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 应用CPU线程设置（仅在 CPU 设备时）
    if device.type == 'cpu' and args.num_threads is not None and args.num_threads > 0:
        try:
            torch.set_num_threads(args.num_threads)
            torch.set_num_interop_threads(max(1, min(args.num_threads, 2)))
            print(f"Set torch threads: intra-op={args.num_threads}, inter-op={max(1, min(args.num_threads, 2))}")
        except Exception as e:
            print(f"Warning: failed to set torch threads: {e}")


    # 如果仅测试划分逻辑
    if args.split_test_only:
        _run_split_test(args, cfg)
        return

    # LOCO评估优先于其他模式
    if args.leave_one_cancer_out:
        return run_loco_training(args, cfg, device)

    # 新增：当设置 --kfold_cancer 时，执行逐折训练并保存每折产物与汇总CSV
    if args.kfold_cancer is not None and args.kfold_cancer > 0:
        return run_kfold_training(args, cfg, device)

    # 默认单次训练模式（HPO/复评请使用 train_hpo.py）
    return run_single_training(args, cfg, device)

def prepare_graphs(args, cfg, save_preprocessor_dir=None):
    """构建PyG图，返回 train_graphs, val_graphs, in_dim, n_domains_batch, n_domains_cancer
    支持：当 --stratify_by_cancer 启用时，采用癌种分层划分（每癌种1张为验证）。
    """
    data = np.load(args.train_npz, allow_pickle=True)
    Xs = list(data['Xs']); ys = list(data['ys']); xys = list(data['xys']); slide_ids = list(data['slide_ids'])
    gene_names = list(data['gene_names'])
    slides = [{'X':X,'y':y,'xy':xy,'slide_id':sid} for X,y,xy,sid in zip(Xs, ys, xys, slide_ids)]

    # 域标签准备
    present_ids = [str(sid) for sid in slide_ids]
    id2type, type2present = _build_type_to_present_ids(present_ids)
    _, _, id2batch, _ = _load_cancer_labels()
    # batch域：来自 cancer_sample_labels.csv 的 Batch_id
    batch_fallback_cache = set()
    batch_ids = [_resolve_batch_id(sid, id2batch, batch_fallback_cache) for sid in present_ids]
    batch_to_idx = {bid: i for i, bid in enumerate(sorted(set(batch_ids)))}
    # cancer域：每个癌种一个域
    cancer_types_sorted = sorted(type2present.keys())
    cancer_to_idx = {ct:i for i, ct in enumerate(cancer_types_sorted)}

    # 计算实际使用的 n_hvg（'all' -> 全部基因数）
    _n_hvg_cfg = cfg.get('n_hvg', 'all')
    n_hvg_val = len(gene_names) if (isinstance(_n_hvg_cfg, str) and _n_hvg_cfg.lower() == 'all') else int(_n_hvg_cfg)

    pp = Preprocessor(n_hvg=n_hvg_val, pca_dim=cfg['pca_dim'], use_pca=cfg.get('use_pca', True))
    pp.fit(slides, gene_names)
    if save_preprocessor_dir is not None:
        pp.save(save_preprocessor_dir)

    pyg_graphs = []
    for s in slides:
        Xp = pp.transform(s['X'], gene_names)
        data_g = assemble_pyg(Xp, s['xy'], s['y'], cfg)
        data_g.slide_id = str(s['slide_id'])
        # 注入域标签（图级）
        batch_id = _resolve_batch_id(str(s['slide_id']), id2batch, batch_fallback_cache)
        data_g.bat_dom = torch.tensor(batch_to_idx[batch_id], dtype=torch.long)
        ctype = id2type.get(str(s['slide_id']))
        data_g.cancer_dom = torch.tensor(cancer_to_idx[ctype], dtype=torch.long)
        pyg_graphs.append(data_g)

    in_dim = pyg_graphs[0].x.shape[1]

    n_domains_batch = len(batch_to_idx) if (cfg.get('use_domain_adv_slide') if cfg.get('use_domain_adv_slide') is not None else cfg['use_domain_adv']) else None
    n_domains_cancer = len(cancer_to_idx) if cfg.get('use_domain_adv_cancer', False) else None

    # 默认：启用分层，按比例划分验证集；若未启用分层，则进行简单划分
    if getattr(args, 'stratify_by_cancer', True):
        rng = random.Random(args.split_seed)
        val_ratio = getattr(args, 'val_ratio', 0.2)
        train_ids, val_ids, _ = _stratified_single_split(present_ids, rng, val_ratio=val_ratio)
        id2graph = {str(g.slide_id): g for g in pyg_graphs}
        train_graphs = [id2graph[sid] for sid in train_ids]
        val_graphs = [id2graph[sid] for sid in val_ids]
    else:
        # 简单划分：最后一个切片作为验证集（保持向后兼容）
        train_graphs = pyg_graphs[:-1]
        val_graphs = pyg_graphs[-1:]
        train_ids = [str(g.slide_id) for g in train_graphs]
        val_ids = [str(val_graphs[0].slide_id)] if val_graphs else []

    setattr(args, '_train_ids', list(train_ids))
    setattr(args, '_val_ids', list(val_ids))

    return train_graphs, val_graphs, in_dim, n_domains_batch, n_domains_cancer


def _build_external_val_graphs(args, cfg, preprocessor_dir):
    """构建外部验证集图列表（单切片NPZ目录）。"""
    if not args.val_sample_dir:
        return []
    val_dir = os.path.abspath(args.val_sample_dir)
    if not os.path.isdir(val_dir):
        print(f"[Warn] val_sample_dir not found: {val_dir}")
        return []
    import glob
    npz_files = sorted(glob.glob(os.path.join(val_dir, '*.npz')))
    if not npz_files:
        print(f"[Warn] No NPZ files found in val_sample_dir: {val_dir}")
        return []

    try:
        pp = Preprocessor.load(preprocessor_dir)
    except Exception as e:
        print(f"[Warn] Failed to load preprocessor from {preprocessor_dir}: {e}")
        return []

    # 外部验证仅用于分类指标计算，不注入域标签

    graphs = []
    for npz_path in npz_files:
        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"[Warn] Failed to load {npz_path}: {e}")
            continue
        if not {'X', 'xy', 'gene_names', 'y'}.issubset(set(d.files)):
            print(f"[Warn] Missing required keys in {npz_path}. Require X, xy, gene_names, y.")
            continue
        X = d['X']
        xy = d['xy']
        y = d['y']
        gene_names = list(d['gene_names'])
        sample_id = str(d.get('sample_id', Path(npz_path).stem))

        Xp = pp.transform(X, gene_names)
        g = assemble_pyg(Xp, xy, y, cfg)
        g.slide_id = sample_id

        graphs.append(g)
    return graphs


def train_and_validate(
    train_graphs,
    val_graphs,
    in_dim,
    n_domains_batch,
    n_domains_cancer,
    cfg,
    device,
    num_workers=0,
    report_cb=None,
    external_val_graphs=None,
    progress_desc=None,
    progress_leave=False,
):
    """封装单次训练+验证，返回(best_metrics, hist_dict, best_state_dict)
    report_cb(epoch, metrics) 可选用于HPO报告。
    """
    model = STOnco_Classifier(
        in_dim=in_dim,
        hidden=cfg['hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
        use_domain_adv=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains=(n_domains_batch if (cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']) else None), domain_hidden=64,
        use_domain_adv_slide=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains_slide=n_domains_batch,
        use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
    )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
    cel = nn.CrossEntropyLoss()

    train_loader = PyGDataLoader(train_graphs, batch_size=cfg['batch_size_graphs'], shuffle=True, num_workers=num_workers)
    val_loader = PyGDataLoader(val_graphs, batch_size=1, shuffle=False, num_workers=max(0, min(num_workers, 2)))
    extra_val_graphs = list(external_val_graphs) if external_val_graphs else []

    def _compute_losses(out, batch):
        logits = out['logits']
        mask = batch.y >= 0
        if mask.sum() > 0:
            loss_task = bce(logits[mask], batch.y[mask].float())
        else:
            loss_task = torch.tensor(0.0, device=device)
        total = loss_task
        loss_batch = torch.tensor(0.0, device=device)
        loss_cancer = torch.tensor(0.0, device=device)
        if out.get('dom_logits_slide', None) is not None and hasattr(batch, 'bat_dom'):
            if n_domains_batch is None:
                raise ValueError('n_domains_batch is None while batch domain logits are enabled.')
            dom_min = int(batch.bat_dom.min().item())
            dom_max = int(batch.bat_dom.max().item())
            if dom_min < 0 or dom_max >= int(n_domains_batch):
                raise ValueError(
                    f"[DomainCheck] batch_dom out of range: min={dom_min}, max={dom_max}, "
                    f"n_domains_batch={n_domains_batch}, slide_ids={getattr(batch, 'slide_id', 'NA')}"
                )
            loss_dom_batch = cel(out['dom_logits_slide'], batch.bat_dom)
            loss_batch = cfg['lambda_slide'] * loss_dom_batch
            total = total + loss_batch
        if out.get('dom_logits_cancer', None) is not None and hasattr(batch, 'cancer_dom'):
            if n_domains_cancer is None:
                raise ValueError('n_domains_cancer is None while cancer domain logits are enabled.')
            dom_min = int(batch.cancer_dom.min().item())
            dom_max = int(batch.cancer_dom.max().item())
            if dom_min < 0 or dom_max >= int(n_domains_cancer):
                raise ValueError(
                    f"[DomainCheck] cancer_dom out of range: min={dom_min}, max={dom_max}, "
                    f"n_domains_cancer={n_domains_cancer}, slide_ids={getattr(batch, 'slide_id', 'NA')}"
                )
            loss_dom_cancer = cel(out['dom_logits_cancer'], batch.cancer_dom)
            loss_cancer = cfg['lambda_cancer'] * loss_dom_cancer
            total = total + loss_cancer
        return total, loss_task, loss_batch, loss_cancer

    best = {'auroc': -1, 'accuracy': -1, 'state': None, 'epoch': -1, 'macro_f1': float('nan'), 'auprc': float('nan')}
    patience = 0
    early_patience = cfg.get('early_patience', None)
    early_stop_enabled = early_patience is not None and early_patience > 0
    if not early_stop_enabled:
        print('[Info] Early stopping disabled')

    hist = {
        'avg_total_loss': [],
        'avg_task_loss': [],
        'avg_batch_domain_loss': [],
        'avg_cancer_domain_loss': [],
        'var_risk': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'val_auroc': [],
        'val_auprc': [],
    }

    epoch_iter = range(1, cfg['epochs'] + 1)
    if progress_desc:
        epoch_iter = tqdm(epoch_iter, desc=progress_desc, leave=progress_leave)
    for epoch in epoch_iter:
        model.train()
        tot_total = 0.0
        tot_task = 0.0
        tot_batch = 0.0
        tot_cancer = 0.0
        batch_losses = []
        num_batches = 0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(
                batch.x,
                batch.edge_index,
                batch=getattr(batch, 'batch', None),
                edge_weight=getattr(batch, 'edge_weight', None),
                domain_lambda=cfg['lambda_slide'],
                lambda_slide=cfg['lambda_slide'],
                lambda_cancer=cfg['lambda_cancer'],
            )
            loss_total, loss_task, loss_batch, loss_cancer = _compute_losses(out, batch)
            loss_total.backward()
            opt.step()

            tot_total += float(loss_total.item())
            tot_task += float(loss_task.item())
            tot_batch += float(loss_batch.item())
            tot_cancer += float(loss_cancer.item())
            batch_losses.append(float(loss_total.item()))
            num_batches += 1

            with torch.no_grad():
                probs = torch.sigmoid(out['logits'])
                preds = (probs > 0.5).long()
                mask = batch.y >= 0
                if mask.sum() > 0:
                    train_correct += int((preds[mask] == batch.y[mask]).sum().item())
                    train_total += int(mask.sum().item())

        avg_total = tot_total / max(1, num_batches)
        avg_task = tot_task / max(1, num_batches)
        avg_batch = tot_batch / max(1, num_batches)
        avg_cancer = tot_cancer / max(1, num_batches)
        var_risk = float(np.mean([(v - avg_total) ** 2 for v in batch_losses])) if batch_losses else float('nan')
        train_accuracy = float(train_correct / train_total) if train_total > 0 else float('nan')

        hist['avg_total_loss'].append(avg_total)
        hist['avg_task_loss'].append(avg_task)
        hist['avg_batch_domain_loss'].append(avg_batch)
        hist['avg_cancer_domain_loss'].append(avg_cancer)
        hist['var_risk'].append(var_risk)
        hist['train_accuracy'].append(train_accuracy)

        # 验证（内部 + 外部）
        model.eval()
        val_logits_list = []
        val_y_list = []
        per_slide_acc = []
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                out_v = model(
                    vb.x,
                    vb.edge_index,
                    batch=getattr(vb, 'batch', None),
                    edge_weight=getattr(vb, 'edge_weight', None),
                    domain_lambda=cfg['lambda_slide'],
                    lambda_slide=cfg['lambda_slide'],
                    lambda_cancer=cfg['lambda_cancer'],
                )
                logits_v = out_v['logits']
                val_logits_list.append(logits_v.cpu())
                val_y_list.append(vb.y.cpu())
                m_slide = eval_logits(logits_v, vb.y)
                per_slide_acc.append(m_slide.get('accuracy', float('nan')))

            for g in extra_val_graphs:
                vg = g.to(device)
                out_v = model(
                    vg.x,
                    vg.edge_index,
                    batch=getattr(vg, 'batch', None),
                    edge_weight=getattr(vg, 'edge_weight', None),
                    domain_lambda=cfg['lambda_slide'],
                    lambda_slide=cfg['lambda_slide'],
                    lambda_cancer=cfg['lambda_cancer'],
                )
                logits_v = out_v['logits']
                val_logits_list.append(logits_v.cpu())
                val_y_list.append(vg.y.cpu())
                m_slide = eval_logits(logits_v, vg.y)
                per_slide_acc.append(m_slide.get('accuracy', float('nan')))

        if val_logits_list and val_y_list:
            val_logits = torch.cat(val_logits_list, dim=0)
            val_y = torch.cat(val_y_list, dim=0)
            m = eval_logits(val_logits, val_y)
        else:
            m = {'auroc': float('nan'), 'auprc': float('nan'), 'accuracy': float('nan'), 'macro_f1': float('nan')}

        val_accuracy = float(np.nanmean(per_slide_acc)) if per_slide_acc else float('nan')
        hist['val_accuracy'].append(val_accuracy)
        hist['val_macro_f1'].append(m.get('macro_f1', float('nan')))
        hist['val_auroc'].append(m.get('auroc', float('nan')))
        hist['val_auprc'].append(m.get('auprc', float('nan')))

        metrics = {
            'accuracy': val_accuracy,
            'macro_f1': m.get('macro_f1', float('nan')),
            'auroc': m.get('auroc', float('nan')),
            'auprc': m.get('auprc', float('nan')),
        }
        if report_cb is not None:
            try:
                report_cb(epoch, metrics)
            except Exception:
                pass

        if progress_desc:
            try:
                epoch_iter.set_postfix(
                    train_loss=f'{avg_total:.3f}',
                    val_acc=f'{val_accuracy:.3f}'
                )
            except Exception:
                pass

        if val_accuracy > best['accuracy']:
            best = {
                'auroc': metrics['auroc'],
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'auprc': metrics['auprc'],
                'state': copy.deepcopy(model.state_dict()),
                'epoch': epoch,
            }
            patience = 0
        else:
            patience += 1
            if early_stop_enabled and patience >= early_patience:
                break
    return best, hist, best['state']


def _save_loss_components_csv(hist, out_dir):
    n_epochs = len(hist.get('avg_total_loss', []))
    if n_epochs == 0:
        return None
    df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'avg_total_loss': hist.get('avg_total_loss', [float('nan')] * n_epochs),
        'avg_task_loss': hist.get('avg_task_loss', [float('nan')] * n_epochs),
        'Var_risk': hist.get('var_risk', [float('nan')] * n_epochs),
        'avg_cancer_domain_loss': hist.get('avg_cancer_domain_loss', [float('nan')] * n_epochs),
        'avg_batch_domain_loss': hist.get('avg_batch_domain_loss', [float('nan')] * n_epochs),
        'train_accuracy': hist.get('train_accuracy', [float('nan')] * n_epochs),
        'val_accuracy': hist.get('val_accuracy', [float('nan')] * n_epochs),
        'val_macro_f1': hist.get('val_macro_f1', [float('nan')] * n_epochs),
        'val_auroc': hist.get('val_auroc', [float('nan')] * n_epochs),
        'val_auprc': hist.get('val_auprc', [float('nan')] * n_epochs),
    })
    csv_path = os.path.join(out_dir, 'loss_components.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    return csv_path


def _plot_train_metrics(hist, out_dir):
    n_epochs = len(hist.get('avg_total_loss', []))
    if n_epochs == 0:
        return None, None
    epochs = list(range(1, n_epochs + 1))
    line_color = '#08283D'
    do_smooth = n_epochs > 100
    smooth_window = 10
    raw_alpha = 0.35
    raw_lw = 0.6
    smooth_lw = 1.4

    def _plot_neurips(ax, values, title):
        series = pd.Series(values, dtype='float64')
        if do_smooth:
            ax.plot(
                epochs,
                series.values,
                color=line_color,
                linewidth=raw_lw,
                alpha=raw_alpha,
            )
            smooth = series.rolling(window=smooth_window, min_periods=1).mean()
            ax.plot(
                epochs,
                smooth.values,
                color=line_color,
                linewidth=smooth_lw,
            )
        else:
            ax.plot(
                epochs,
                series.values,
                color=line_color,
                linewidth=smooth_lw,
            )

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    # 1) 训练损失图（2x3）
    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    metrics_train = [
        ('avg_total_loss', 'avg_total_loss'),
        ('avg_task_loss', 'avg_task_loss'),
        ('var_risk', 'Var_risk'),
        ('avg_cancer_domain_loss', 'avg_cancer_domain_loss'),
        ('avg_batch_domain_loss', 'avg_batch_domain_loss'),
        ('train_accuracy', 'train_accuracy'),
    ]
    for ax, (key, title) in zip(axes1.flatten(), metrics_train):
        values = hist.get(key, [float('nan')] * n_epochs)
        _plot_neurips(ax, values, title)
    for ax in axes1[0]:
        ax.tick_params(labelbottom=True)
    fig1.tight_layout()
    out_train_svg = os.path.join(out_dir, 'train_loss.svg')
    fig1.savefig(out_train_svg, format='svg', dpi=150)
    plt.close(fig1)

    # 2) 验证指标图（2x2）
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    metrics_val = [
        ('val_accuracy', 'val_accuracy'),
        ('val_macro_f1', 'val_macro_f1'),
        ('val_auroc', 'val_auroc'),
        ('val_auprc', 'val_auprc'),
    ]
    for ax, (key, title) in zip(axes2.flatten(), metrics_val):
        values = hist.get(key, [float('nan')] * n_epochs)
        _plot_neurips(ax, values, title)
    for ax in axes2[0]:
        ax.tick_params(labelbottom=True)
    fig2.tight_layout()
    out_val_svg = os.path.join(out_dir, 'train_val_metrics.svg')
    fig2.savefig(out_val_svg, format='svg', dpi=150)
    plt.close(fig2)
    return out_train_svg, out_val_svg


def run_single_training(args, cfg, device):
    """原始的单次训练逻辑（现支持双域对抗）"""
    
    train_graphs, val_graphs, in_dim, n_domains_batch, n_domains_cancer = prepare_graphs(args, cfg, save_preprocessor_dir=args.artifacts_dir)

    external_val_graphs = _build_external_val_graphs(args, cfg, args.artifacts_dir)

    best, hist, best_state = train_and_validate(
        train_graphs,
        val_graphs,
        in_dim,
        n_domains_batch,
        n_domains_cancer,
        cfg,
        device,
        num_workers=args.num_workers,
        external_val_graphs=external_val_graphs,
        progress_desc='Train',
        progress_leave=True,
    )

    # 保存最优模型
    model = STOnco_Classifier(
        in_dim=in_dim,
        hidden=cfg['hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
        use_domain_adv=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains=(n_domains_batch if (cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']) else None), domain_hidden=64,
        use_domain_adv_slide=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains_slide=n_domains_batch,
        use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
    )
    model.load_state_dict(best_state)
    save_model(model, args.artifacts_dir)
    train_ids = list(getattr(args, '_train_ids', []))
    val_ids = list(getattr(args, '_val_ids', []))
    meta = {
        'cfg': cfg,
        'best_epoch': best['epoch'],
        'val_ids': val_ids,
        'train_ids': train_ids,
        'metrics': {
            'auroc': best.get('auroc', float('nan')),
            'auprc': best.get('auprc', float('nan')),
            'accuracy': best.get('accuracy', float('nan')),
            'macro_f1': best.get('macro_f1', float('nan')),
        }
    }
    save_json(meta, os.path.join(args.artifacts_dir, 'meta.json'))
    print('Saved artifacts to', args.artifacts_dir)

    # 保存训练与验证曲线
    try:
        os.makedirs(args.artifacts_dir, exist_ok=True)
        if getattr(args, 'save_train_curves', 1):
            out_train_svg, out_val_svg = _plot_train_metrics(hist, args.artifacts_dir)
            if out_train_svg:
                print('Saved training loss figure to', out_train_svg)
            if out_val_svg:
                print('Saved validation metrics figure to', out_val_svg)

        if getattr(args, 'save_loss_components', 0):
            csv_path = _save_loss_components_csv(hist, args.artifacts_dir)
            if csv_path:
                print('Saved loss components to', csv_path)
    except Exception as e:
        print('Warning: failed to save training metrics figures:', e)

    # 新增：训练结束后计算并保存总体基因重要性（不分癌种）
    if getattr(args, 'explain_saliency', False):
        try:
            print('[Explain] Start computing gene importance using method:', args.explain_method)
            # 加载预处理器以获取HVG、PCA与Scaler
            pp = Preprocessor.load(args.artifacts_dir)
            use_pca = bool(getattr(pp, 'use_pca', True) and getattr(pp, 'pca', None) is not None)
            feat_dim_gene = (int(pp.pca.n_components_) if use_pca else len(pp.hvg))

            model = model.to(device)
            model.eval()

            def compute_graph_attr(g):
                # 输入张量拷贝并开启梯度
                x = g.x.to(device)
                edge_index = g.edge_index.to(device)
                edge_weight = getattr(g, 'edge_weight', None)
                if edge_weight is not None:
                    edge_weight = edge_weight.to(device)
                x_in = x.detach().clone()
                # 基线：仅对基因维度设为0，PE部分保持原值
                baseline = x_in.detach().clone()
                baseline[:, :feat_dim_gene] = 0.0
                delta = (x_in - baseline)

                if args.explain_method == 'ig':
                    steps = max(1, int(getattr(args, 'ig_steps', 50)))
                    grads_sum = torch.zeros_like(x_in[:, :feat_dim_gene])
                    for s in range(1, steps + 1):
                        alpha = float(s) / float(steps)
                        x_step = baseline + delta * alpha
                        x_step = x_step.detach().requires_grad_(True)
                        out = model(x_step, edge_index, batch=getattr(g, 'batch', None), edge_weight=edge_weight)
                        logits = out['logits']
                        loss = logits.sum()
                        grad = torch.autograd.grad(loss, x_step, retain_graph=False)[0]
                        grads_sum = grads_sum + grad[:, :feat_dim_gene].detach()
                    avg_grads = grads_sum / float(steps)
                    attr_feat = (delta[:, :feat_dim_gene] * avg_grads)  # (N, F_gene)
                    # 归约为图级：按节点取平均的绝对贡献
                    attr_feat_graph = attr_feat.abs().mean(dim=0)  # (F_gene,)
                else:  # saliency
                    x_sal = x_in.detach().clone().requires_grad_(True)
                    out = model(x_sal, edge_index, batch=getattr(g, 'batch', None), edge_weight=edge_weight)
                    logits = out['logits']
                    loss = logits.sum()
                    grad = torch.autograd.grad(loss, x_sal, retain_graph=False)[0]
                    sal = grad[:, :feat_dim_gene]
                    attr_feat_graph = sal.abs().mean(dim=0)

                # 回投到HVG基因空间
                if use_pca:
                    W = torch.from_numpy(pp.pca.components_.astype(np.float32)).to(attr_feat_graph.device)  # (pca_dim, n_hvg)
                    attr_z = attr_feat_graph @ W  # (n_hvg,)
                else:
                    attr_z = attr_feat_graph  # 已经在z-space
                # 还原到原始基因尺度（链式法则：dz/dx = 1/scale）
                scale = torch.from_numpy(pp.scaler.scale_.astype(np.float32)).to(attr_z.device)
                scale = torch.clamp(scale, min=1e-6)
                attr_gene = attr_z / scale  # (n_hvg,)
                return attr_gene.detach().cpu()

            # 在训练图上聚合
            per_graph_attrs = []
            for g in train_graphs:
                try:
                    per_graph_attrs.append(compute_graph_attr(g))
                except Exception as e:
                    print(f"[Explain] Warning: failed on slide {getattr(g, 'slide_id', 'NA')}: {e}")
            if len(per_graph_attrs) == 0:
                print('[Explain] No graph-level attributions computed; skip writing CSV.')
            else:
                A = torch.stack(per_graph_attrs, dim=0)  # (G, n_hvg)
                agg = A.mean(dim=0).numpy()
                import pandas as pd
                df_attr = pd.DataFrame({'gene': pp.hvg, 'attr': agg})
                out_csv = os.path.join(args.artifacts_dir, 'per_gene_saliency.csv')
                df_attr.to_csv(out_csv, index=False)
                print('[Explain] Saved overall gene importance to', out_csv)
        except Exception as e:
            print('[Explain] Failed to compute gene importance:', e)




def run_kfold_training(args, cfg, device):
    """基于癌种的K折训练：
    - 先构建所有切片的图（与 prepare_graphs 类似），避免重复预处理
    - 生成K个癌种分层的fold
    - 逐折训练，产物保存到 artifacts_dir 同级目录的 kfold_val/fold_{i}/
    - 在 artifacts_dir 同级目录的 kfold_val/ 下输出 kfold_summary.csv
    """
    # 读取数据并构建图
    data = np.load(args.train_npz, allow_pickle=True)
    Xs = list(data['Xs']); ys = list(data['ys']); xys = list(data['xys']); slide_ids = list(data['slide_ids'])
    gene_names = list(data['gene_names'])
    slides = [{'X': X, 'y': y, 'xy': xy, 'slide_id': sid} for X, y, xy, sid in zip(Xs, ys, xys, slide_ids)]

    present_ids = [str(sid) for sid in slide_ids]
    id2type, type2present = _build_type_to_present_ids(present_ids)
    _, _, id2batch, _ = _load_cancer_labels()
    batch_fallback_cache = set()
    batch_ids = [_resolve_batch_id(sid, id2batch, batch_fallback_cache) for sid in present_ids]
    batch_to_idx = {bid: i for i, bid in enumerate(sorted(set(batch_ids)))}
    cancer_types_sorted = sorted(type2present.keys())
    cancer_to_idx = {ct:i for i, ct in enumerate(cancer_types_sorted)}

    # 计算实际使用的 n_hvg（'all' -> 全部基因数）
    _n_hvg_cfg = cfg.get('n_hvg', 'all')
    n_hvg_val = len(gene_names) if (isinstance(_n_hvg_cfg, str) and _n_hvg_cfg.lower() == 'all') else int(_n_hvg_cfg)

    pp = Preprocessor(n_hvg=n_hvg_val, pca_dim=cfg['pca_dim'], use_pca=cfg.get('use_pca', True))
    pp.fit(slides, gene_names)

    pyg_graphs = []
    for s in slides:
        Xp = pp.transform(s['X'], gene_names)
        data_g = assemble_pyg(Xp, s['xy'], s['y'], cfg)
        data_g.slide_id = str(s['slide_id'])
        batch_id = _resolve_batch_id(str(s['slide_id']), id2batch, batch_fallback_cache)
        data_g.bat_dom = torch.tensor(batch_to_idx[batch_id], dtype=torch.long)
        ctype = id2type.get(str(s['slide_id']))
        data_g.cancer_dom = torch.tensor(cancer_to_idx[ctype], dtype=torch.long)
        pyg_graphs.append(data_g)

    in_dim = pyg_graphs[0].x.shape[1]

    n_domains_batch = len(batch_to_idx) if (cfg.get('use_domain_adv_slide') if cfg.get('use_domain_adv_slide') is not None else cfg['use_domain_adv']) else None
    n_domains_cancer = len(cancer_to_idx) if cfg.get('use_domain_adv_cancer', False) else None

    # 生成K个fold（按比例 + 保底每癌种1张）
    rng = random.Random(args.split_seed)
    folds, present_id2type = _k_random_combinations(present_ids, int(args.kfold_cancer), rng, val_ratio=args.val_ratio)

    id2graph = {str(g.slide_id): g for g in pyg_graphs}

    # 汇总结果
    results = []
    # 将 kfold_val 放置到 artifacts_dir 的同级目录
    parent_dir = os.path.abspath(os.path.join(args.artifacts_dir, os.pardir))
    base_kfold_dir = os.path.join(parent_dir, 'kfold_val')
    os.makedirs(base_kfold_dir, exist_ok=True)
    print(f"[KFold] Base directory: {base_kfold_dir}")

    total_folds = len(folds)
    for i, (train_ids, val_ids) in enumerate(tqdm(folds, desc='KFold', total=total_folds), start=1):
        fold_dir = os.path.join(base_kfold_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"[KFold] Created fold directory: {fold_dir}")

        # 保存本折的预处理器产物（此处pp基于全体拟合，若需严格无泄漏，可改为每折重拟合）
        try:
            pp.save(fold_dir)
            print(f"[KFold] Saved preprocessor artifacts to {fold_dir}")
        except Exception:
            pass

        train_graphs = [id2graph[sid] for sid in train_ids]
        val_graphs = [id2graph[sid] for sid in val_ids]

        external_val_graphs = _build_external_val_graphs(args, cfg, fold_dir)
        best, hist, best_state = train_and_validate(
            train_graphs,
            val_graphs,
            in_dim,
            n_domains_batch,
            n_domains_cancer,
            cfg,
            device,
            num_workers=args.num_workers,
            external_val_graphs=external_val_graphs,
            progress_desc=f'Fold {i}/{total_folds}',
            progress_leave=False,
        )

        if getattr(args, 'save_train_curves', 1):
            try:
                out_train_svg, out_val_svg = _plot_train_metrics(hist, fold_dir)
                if out_train_svg:
                    print(f'[KFold] Saved training loss figure to {out_train_svg}')
                if out_val_svg:
                    print(f'[KFold] Saved validation metrics figure to {out_val_svg}')
            except Exception as e:
                print(f'[KFold] Warning: failed to save training curves: {e}')

        # 新增：如果启用save_loss_components，保存Loss组件CSV
        if getattr(args, 'save_loss_components', 0):
            try:
                csv_path = _save_loss_components_csv(hist, fold_dir)
                if csv_path:
                    print(f'[KFold] Saved loss components to {csv_path}')
            except Exception as e:
                print(f'[KFold] Warning: failed to save loss components CSV: {e}')

        # 保存最优模型及元信息
        model = STOnco_Classifier(
            in_dim=in_dim,
            hidden=cfg['hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
            use_domain_adv=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains=(n_domains_batch if (cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']) else None), domain_hidden=64,
            use_domain_adv_slide=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains_slide=n_domains_batch,
            use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
        )
        model.load_state_dict(best_state)
        save_model(model, fold_dir)
        model_path = os.path.join(fold_dir, 'model.pt')
        if os.path.exists(model_path):
            print(f"[KFold] Saved model to {model_path}")
        else:
            print(f"[KFold] Saved model to {fold_dir} (model.pt)")
        meta = {
            'cfg': cfg,
            'best_epoch': best['epoch'],
            'val_ids': val_ids,
            'train_ids': train_ids,
            'metrics': {
                'auroc': best.get('auroc', float('nan')),
                'auprc': best.get('auprc', float('nan')),
                'accuracy': best.get('accuracy', float('nan')),
                'macro_f1': best.get('macro_f1', float('nan')),
            }
        }
        meta_path = os.path.join(fold_dir, 'meta.json')
        save_json(meta, meta_path)
        print(f"[KFold] Saved meta to {meta_path}")

        results.append({
            'fold': i,
            'best_epoch': best['epoch'],
            'auroc': best.get('auroc', float('nan')),
            'auprc': best.get('auprc', float('nan')),
            'accuracy': best.get('accuracy', float('nan')),
            'macro_f1': best.get('macro_f1', float('nan')),
            'n_train': len(train_ids),
            'n_val': len(val_ids)
        })

    # 写出汇总CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        summary_path = os.path.join(base_kfold_dir, 'kfold_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"Saved kfold summary to {summary_path}")
        # 打印均值
        means = df[['auroc', 'auprc', 'accuracy', 'macro_f1']].mean(numeric_only=True)
        print('K-fold mean metrics:', {k: float(v) for k, v in means.items()})
    except Exception as e:
        print(f"Warning: failed to write kfold summary CSV: {e}")

    return {
        'fold_results': results,
        'artifacts_dir': args.artifacts_dir
    }


def run_loco_training(args, cfg, device):
    """留一癌种评估：对每个癌种ct，将其全部切片作为验证集，其余作为训练集；每个ct单独训练一个模型。
    预处理（PCA/Scaler等）严格在训练集拟合，避免泄漏。
    产物目录：在 artifacts_dir 同级目录创建 loco_eval/ct/ 子目录。
    """
    data = np.load(args.train_npz, allow_pickle=True)
    Xs = list(data['Xs']); ys = list(data['ys']); xys = list(data['xys']); slide_ids = list(map(str, list(data['slide_ids'])))
    gene_names = list(data['gene_names'])

    id2type, type2present = _build_type_to_present_ids(slide_ids)
    _, _, id2batch, _ = _load_cancer_labels()
    cancer_types = sorted(type2present.keys())

    parent_dir = os.path.abspath(os.path.join(args.artifacts_dir, os.pardir))
    base_loco_dir = os.path.join(parent_dir, 'loco_eval')
    os.makedirs(base_loco_dir, exist_ok=True)

    results = []
    # 新增：跨癌种的逐样本记录（顶层汇总输出）
    per_slide_rows = []

    # 将原始slides打包便于按索引选择
    slides_all = [{'X': X, 'y': y, 'xy': xy, 'slide_id': sid} for X, y, xy, sid in zip(Xs, ys, xys, slide_ids)]

    for ct in tqdm(cancer_types, desc='LOCO', total=len(cancer_types)):
        val_ids = sorted(type2present[ct])
        train_ids = sorted([sid for sid in slide_ids if sid not in set(val_ids)])
        if not train_ids or not val_ids:
            print(f"[LOCO] Skip cancer_type={ct} due to empty train/val partition.")
            continue
        loco_dir = os.path.join(base_loco_dir, ct)
        os.makedirs(loco_dir, exist_ok=True)

        # 拟合预处理器（仅训练集）
        slides_train = [s for s in slides_all if s['slide_id'] in train_ids]

        # 计算实际使用的 n_hvg（'all' -> 全部基因数）
        _n_hvg_cfg = cfg.get('n_hvg', 'all')
        n_hvg_val = len(gene_names) if (isinstance(_n_hvg_cfg, str) and _n_hvg_cfg.lower() == 'all') else int(_n_hvg_cfg)

        pp = Preprocessor(n_hvg=n_hvg_val, pca_dim=cfg['pca_dim'], use_pca=cfg.get('use_pca', True))
        pp.fit(slides_train, gene_names)
        try:
            pp.save(loco_dir)
        except Exception:
            pass

        # 构建图并注入域标签
        # 域索引基于全部出现的batch/cancer，以保持head维度稳定
        batch_fallback_cache = set()
        batch_ids = [_resolve_batch_id(sid, id2batch, batch_fallback_cache) for sid in slide_ids]
        batch_to_idx = {bid: i for i, bid in enumerate(sorted(set(batch_ids)))}
        cancer_to_idx = {c:i for i, c in enumerate(cancer_types)}

        def build_graph_list(id_list):
            graphs = []
            for s in slides_all:
                if s['slide_id'] not in id_list:
                    continue
                Xp = pp.transform(s['X'], gene_names)
                g = assemble_pyg(Xp, s['xy'], s['y'], cfg)
                g.slide_id = str(s['slide_id'])
                batch_id = _resolve_batch_id(str(s['slide_id']), id2batch, batch_fallback_cache)
                g.bat_dom = torch.tensor(batch_to_idx[batch_id], dtype=torch.long)
                g.cancer_dom = torch.tensor(cancer_to_idx[id2type[str(s['slide_id'])]], dtype=torch.long)
                graphs.append(g)
            return graphs

        train_graphs = build_graph_list(train_ids)
        val_graphs = build_graph_list(val_ids)

        in_dim = train_graphs[0].x.shape[1]
        n_domains_batch = len(batch_to_idx) if (cfg.get('use_domain_adv_slide') if cfg.get('use_domain_adv_slide') is not None else cfg['use_domain_adv']) else None
        n_domains_cancer = len(cancer_to_idx) if cfg.get('use_domain_adv_cancer', False) else None

        external_val_graphs = _build_external_val_graphs(args, cfg, loco_dir)
        best, hist, best_state = train_and_validate(
            train_graphs,
            val_graphs,
            in_dim,
            n_domains_batch,
            n_domains_cancer,
            cfg,
            device,
            num_workers=args.num_workers,
            external_val_graphs=external_val_graphs,
            progress_desc=f'LOCO {ct}',
            progress_leave=False,
        )

        if getattr(args, 'save_train_curves', 1):
            try:
                out_train_svg, out_val_svg = _plot_train_metrics(hist, loco_dir)
                if out_train_svg:
                    print(f'[LOCO] Saved training loss figure to {out_train_svg}')
                if out_val_svg:
                    print(f'[LOCO] Saved validation metrics figure to {out_val_svg}')
            except Exception as e:
                print(f'[LOCO] Warning: failed to save training curves: {e}')

        # 新增：如果启用save_loss_components，保存Loss组件CSV
        if getattr(args, 'save_loss_components', 0):
            try:
                csv_path = _save_loss_components_csv(hist, loco_dir)
                if csv_path:
                    print(f'[LOCO] Saved loss components to {csv_path}')
            except Exception as e:
                print(f'[LOCO] Warning: failed to save loss components CSV: {e}')

        # 保存模型与meta
        model = STOnco_Classifier(
            in_dim=in_dim,
            hidden=cfg['hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
            use_domain_adv=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains=(n_domains_batch if (cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']) else None), domain_hidden=64,
            use_domain_adv_slide=(cfg['use_domain_adv_slide'] if cfg['use_domain_adv_slide'] is not None else cfg['use_domain_adv']), n_domains_slide=n_domains_batch,
            use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
        )
        model.load_state_dict(best_state)
        save_model(model, loco_dir)
        meta = {
            'cfg': cfg,
            'best_epoch': best['epoch'],
            'val_ids': val_ids,
            'train_ids': train_ids,
            'metrics': {
                'auroc': best.get('auroc', float('nan')),
                'auprc': best.get('auprc', float('nan')),
                'accuracy': best.get('accuracy', float('nan')),
                'macro_f1': best.get('macro_f1', float('nan')),
            }
        }
        save_json(meta, os.path.join(loco_dir, 'meta.json'))

        # 新增：逐样本（逐 slide）评估，集中写到顶层 loco_per_slide.csv
        try:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                for g in val_graphs:
                    vg = g.to(device)
                    out_v = model(vg.x, vg.edge_index, batch=getattr(vg, 'batch', None), edge_weight=getattr(vg, 'edge_weight', None),
                                   domain_lambda=cfg['lambda_slide'], lambda_slide=cfg['lambda_slide'], lambda_cancer=cfg['lambda_cancer'])
                    logits_v = out_v['logits']
                    m_slide = eval_logits(logits_v, vg.y)
                    y_np = vg.y.detach().cpu().numpy()
                    n_nodes = int(y_np.shape[0])
                    n_pos = int((y_np == 1).sum())
                    n_neg = int((y_np == 0).sum())
                    pos_rate = float(n_pos / n_nodes) if n_nodes > 0 else float('nan')
                    per_slide_rows.append({
                        'cancer_type': ct,
                        'slide_id': str(g.slide_id),
                        'n_nodes': n_nodes,
                        'n_pos': n_pos,
                        'n_neg': n_neg,
                        'pos_rate': pos_rate,
                        'auroc': m_slide.get('auroc', float('nan')),
                        'auprc': m_slide.get('auprc', float('nan')),
                        'accuracy': m_slide.get('accuracy', float('nan')),
                        'macro_f1': m_slide.get('macro_f1', float('nan')),
                        'best_epoch': best['epoch']
                    })
        except Exception as e:
            print(f"[LOCO] Warning: per-slide evaluation failed for cancer_type={ct}: {e}")

        results.append({
            'cancer_type': ct,
            'best_epoch': best['epoch'],
            'auroc': best.get('auroc', float('nan')),
            'auprc': best.get('auprc', float('nan')),
            'accuracy': best.get('accuracy', float('nan')),
            'macro_f1': best.get('macro_f1', float('nan')),
            'n_train': len(train_ids),
            'n_val': len(val_ids)
        })

    # 新增：写出顶层逐样本与“样本均值”汇总
    try:
        if len(per_slide_rows) > 0:
            df_ps = pd.DataFrame(per_slide_rows)
            per_slide_path = os.path.join(base_loco_dir, 'loco_per_slide.csv')
            df_ps.to_csv(per_slide_path, index=False)
            print(f"Saved per-slide metrics to {per_slide_path}")

            # 构建按癌种的样本级均值/标准差汇总，以及 ALL 行
            metrics = ['auroc', 'auprc', 'accuracy', 'macro_f1']
            summary_rows = []
            for ct, sub in df_ps.groupby('cancer_type'):
                row = {
                    'cancer_type': ct,
                    'n_val_slides': int(len(sub))
                }
                for m in metrics:
                    row[f'mean_{m}_slide'] = float(sub[m].mean()) if m in sub else float('nan')
                    # 使用样本标准差（与 pandas 默认一致）；若只有一个样本则 std 为 NaN
                    row[f'std_{m}_slide'] = float(sub[m].std()) if m in sub else float('nan')
                summary_rows.append(row)
            # ALL 行
            all_row = {'cancer_type': 'ALL', 'n_val_slides': int(len(df_ps))}
            for m in metrics:
                all_row[f'mean_{m}_slide'] = float(df_ps[m].mean()) if m in df_ps else float('nan')
                all_row[f'std_{m}_slide'] = float(df_ps[m].std()) if m in df_ps else float('nan')
            summary_rows.append(all_row)

            df_ps_sum = pd.DataFrame(summary_rows)
            per_slide_sum_path = os.path.join(base_loco_dir, 'loco_per_slide_summary.csv')
            df_ps_sum.to_csv(per_slide_sum_path, index=False)
            print(f"Saved per-slide summary to {per_slide_sum_path}")
    except Exception as e:
        print(f"Warning: failed to write per-slide CSVs: {e}")

    # 汇总CSV（保持现有：整体节点合并后的指标，不混入样本均值）
    try:
        df = pd.DataFrame(results)
        summary_path = os.path.join(base_loco_dir, 'loco_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"Saved LOCO summary to {summary_path}")
    except Exception as e:
        print(f"Warning: failed to write LOCO summary CSV: {e}")

    return {'loco_results': results, 'base_dir': base_loco_dir}


def _run_split_test(args, cfg):
    """仅测试划分逻辑：
    - 若 --stratify_by_cancer: 打印分层单折的每癌种训练/验证样本数
    - 若 --kfold_cancer: 生成K个不同组合，打印每折统计
    """
    print('[SplitTest] Loading data and building graphs for split test...')
    data = np.load(args.train_npz, allow_pickle=True)
    slide_ids = list(map(str, list(data['slide_ids'])))

    rng = random.Random(args.split_seed)

    # 统计辅助
    def summarize(ids):
        id2type, type2present = _build_type_to_present_ids(ids)
        cnt = {}
        for s in ids:
            t = id2type.get(str(s), 'UNKNOWN')
            cnt[t] = cnt.get(t, 0) + 1
        return cnt

    if args.stratify_by_cancer and not args.kfold_cancer:
        train_ids, val_ids, _ = _stratified_single_split(slide_ids, rng, val_ratio=args.val_ratio)
        print('[SplitTest] Stratified single split:')
        print('  Total slides:', len(slide_ids))
        print('  Val ratio  :', args.val_ratio)
        print('  Train count:', len(train_ids))
        print('  Val count  :', len(val_ids))
        tr_cnt = summarize(train_ids)
        va_cnt = summarize(val_ids)
        print('  Train per cancer:', tr_cnt)
        print('  Val per cancer  :', va_cnt)
        print('  Val unique slides:', sorted(val_ids))
    elif args.kfold_cancer and args.kfold_cancer > 0:
        folds, _ = _k_random_combinations(slide_ids, args.kfold_cancer, rng, val_ratio=args.val_ratio)
        print(f'[SplitTest] K-fold by cancer with K={len(folds)} (random distinct combinations):')
        for i, (tr, va) in enumerate(folds, 1):
            tr_cnt = summarize(tr); va_cnt = summarize(va)
            print(f'  Fold {i}: Train={len(tr)}, Val={len(va)}')
            print(f'    Train per cancer: {tr_cnt}')
            print(f'    Val per cancer  : {va_cnt}')
            print(f'    Val slides: {sorted(va)}')
    else:
        # 默认行为预览
        print('[SplitTest] Default last-slice split:')
        print('  Total slides:', len(slide_ids))
        if len(slide_ids) > 0:
            print('  Val slide:', slide_ids[-1])
            print('  Train slides:', slide_ids[:-1])
        else:
            print('  No slides found in NPZ.')



if __name__ == '__main__':
    main()
