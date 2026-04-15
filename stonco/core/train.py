import argparse, os, numpy as np, torch, math, itertools, shutil
from stonco.utils.preprocessing import Preprocessor, GraphBuilder, ImagePreprocessor, build_node_features_early_fusion
from .models import STOnco_Classifier
from .wb_potentials import GeneratedSupportMap, GeneratedSupportWBLoss
from .sampler import (
    CancerBalancedBatchSampler,
    build_training_subgraphs,
    normalize_sampler_config,
    summarize_batches,
)
from stonco.utils.utils import normalize_gnn_config, save_model, save_json, load_json
from torch_geometric.data import Data as PyGData, DataLoader as PyGDataLoader
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


def _normalize_save_epoch_checkpoints(raw_value):
    values = []

    def _append(value):
        if value is None:
            return
        if isinstance(value, bool):
            raise ValueError(f'save_epoch_checkpoints must be positive integers, got bool: {value}')
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _append(item)
            return
        if isinstance(value, str):
            s = value.strip()
            if s == '':
                return
            for part in s.split(','):
                part = part.strip()
                if part == '':
                    continue
                _append(part)
            return
        try:
            iv = int(value)
        except Exception as exc:
            raise ValueError(f'save_epoch_checkpoints must contain positive integers, got: {value}') from exc
        if iv <= 0:
            raise ValueError(f'save_epoch_checkpoints must be > 0, got: {iv}')
        values.append(iv)

    _append(raw_value)
    return sorted(set(values))


def _slice_hist_until_epoch(hist, epoch):
    hist = dict(hist or {})
    epoch = int(epoch)
    sliced = {}
    for key, values in hist.items():
        if isinstance(values, pd.Series):
            sliced[key] = values.iloc[:epoch].tolist()
        elif isinstance(values, np.ndarray):
            sliced[key] = values[:epoch].tolist()
        elif isinstance(values, (list, tuple)):
            sliced[key] = list(values[:epoch])
        else:
            sliced[key] = copy.deepcopy(values)
    return sliced


def _copy_epoch_static_artifacts(source_dir, target_dir):
    if source_dir is None:
        return
    filenames = [
        'genes_hvg.txt',
        'scaler.joblib',
        'pca.joblib',
        'img_feature_names.txt',
        'img_scaler.joblib',
        'img_pca.joblib',
    ]
    os.makedirs(target_dir, exist_ok=True)
    for name in filenames:
        src = os.path.join(source_dir, name)
        dst = os.path.join(target_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def _save_extra_epoch_checkpoints(
    model,
    epoch_states,
    cfg,
    out_dir,
    extra_meta=None,
    hist=None,
    save_train_curves=False,
    save_loss_components=False,
    domain_plot_kwargs=None,
    artifact_source_dir=None,
):
    epoch_states = dict(epoch_states or {})
    if len(epoch_states) == 0:
        return []
    extra_meta = dict(extra_meta or {})
    domain_plot_kwargs = dict(domain_plot_kwargs or {})
    artifact_source_dir = artifact_source_dir or out_dir
    ckpt_root = os.path.join(out_dir, 'epoch_checkpoints')
    original_state = copy.deepcopy(model.state_dict())
    saved_epochs = []
    try:
        for epoch in sorted(int(k) for k in epoch_states.keys()):
            item = dict(epoch_states.get(epoch, {}) or {})
            state = item.get('state', None)
            if state is None:
                continue
            metrics = dict(item.get('metrics', {}) or {})
            epoch_dir = os.path.join(ckpt_root, f'epoch_{int(epoch)}')
            model.load_state_dict(state)
            save_model(model, epoch_dir)
            _copy_epoch_static_artifacts(artifact_source_dir, epoch_dir)

            hist_epoch = _slice_hist_until_epoch(hist, epoch) if hist is not None else None
            if hist_epoch is not None and len(hist_epoch.get('avg_total_loss', [])) > 0:
                if save_train_curves:
                    _plot_train_metrics(hist_epoch, epoch_dir)
                    _plot_domain_diagnostics(hist_epoch, epoch_dir, **domain_plot_kwargs)
                    if bool(cfg.get('use_wb_align', False)):
                        _plot_wb_train_metrics(hist_epoch, epoch_dir)
                if save_loss_components:
                    _save_loss_components_csv(hist_epoch, epoch_dir)

            meta = {
                'cfg': cfg,
                'best_epoch': int(epoch),
                'last_epoch': int(epoch),
                'completed_full_epochs': False,
                'saved_checkpoint': 'epoch',
                'saved_epoch': int(epoch),
                'saved_epoch_checkpoints': [int(epoch)],
                'requested_save_epoch_checkpoints': [int(epoch)],
                'metrics': metrics,
                'last_metrics': metrics,
            }
            meta.update(copy.deepcopy(extra_meta))
            save_json(meta, os.path.join(epoch_dir, 'meta.json'))
            print(f'Saved extra epoch artifacts to {epoch_dir}')
            saved_epochs.append(int(epoch))
    finally:
        model.load_state_dict(original_state)
    return saved_epochs

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

def assemble_pyg(Xp_gene, xy, y, cfg, Xp_img=None, img_mask=None):
    gb = GraphBuilder(knn_k=cfg['knn_k'], gaussian_sigma_factor=cfg['gaussian_sigma_factor'])
    edge_index, edge_weight, mean_nd = gb.build_knn(xy)
    # 计算PE（可选使用高斯距离权重）
    if cfg.get('lap_pe_dim', 0) and cfg.get('lap_pe_dim', 0) > 0:
        pe = gb.lap_pe(edge_index, Xp_gene.shape[0], k=cfg['lap_pe_dim'],
                       edge_weight=edge_weight if cfg.get('lap_pe_use_gaussian', False) else None,
                       use_gaussian_weights=cfg.get('lap_pe_use_gaussian', False))
    else:
        pe = None
    x = build_node_features_early_fusion(Xp_gene, cfg, Xp_img=Xp_img, img_mask=img_mask, pe=pe)
    data = PyGData(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index).long(), edge_weight=torch.from_numpy(edge_weight).float(), y=torch.from_numpy(y).long())
    data.num_nodes = x.shape[0]
    data.pos = torch.from_numpy(np.asarray(xy)).float()
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
    parser.add_argument('--sampler_mode', choices=['random', 'cancer_balanced', 'cancer_balanced_subgraph'], default=None, help='训练集sampler模式')
    parser.add_argument('--sampler_k_cancers', type=int, default=None, help='每个batch优先采样的癌种数K')
    parser.add_argument('--sampler_m_per_cancer', type=int, default=None, help='每个癌种优先采样的父切片数M')
    parser.add_argument('--sampler_enforce_distinct_batch', type=int, choices=[0,1], default=None, help='采样时尽量约束同癌种内不同bat_dom (1/0)')
    parser.add_argument('--subgraph_mode', choices=['off', 'static', 'online'], default=None, help='训练子图模式：off/static/online')
    parser.add_argument('--subgraph_target_spots', type=int, default=None, help='静态子图目标spots数')
    parser.add_argument('--subgraph_min_spots', type=int, default=None, help='静态子图最小spots数')
    parser.add_argument('--model', choices=['gatv2', 'sage', 'gcn'], default=None, help='选择GNN主干，默认gatv2')
    parser.add_argument('--heads', type=int, default=None, help='GATv2的多头数（仅对gatv2有效）')
    parser.add_argument('--concat_lap_pe', type=int, choices=[0,1], default=None, help='是否将lapPE拼接至节点特征（1/0）')
    parser.add_argument('--lap_pe_use_gaussian', type=int, choices=[0,1], default=None, help='lapPE是否使用高斯边权（1/0）')
    parser.add_argument('--lap_pe_dim', type=int, default=None, help='lapPE维度（>0表示启用）')
    parser.add_argument('--num_threads', type=int, default=None, help='设置PyTorch计算线程数（CPU模式下限制核心占用）')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader数据加载工作进程数')
    parser.add_argument('--use_pca', type=int, choices=[0, 1], default=None, help='是否使用PCA（1/0）')
    # 新增：图像特征早期融合（方案1）
    parser.add_argument('--use_image_features', type=int, choices=[0, 1], default=None, help='是否启用图像特征早期融合（1/0）')
    parser.add_argument('--img_use_pca', type=int, choices=[0, 1], default=None, help='图像特征是否使用PCA降维（1/0）')
    parser.add_argument('--img_pca_dim', type=int, default=None, help='图像特征PCA维度（默认256，仅当 img_use_pca=1 生效）')
    # 新增：HVG数量控制（支持数值或'all'；默认'all'表示使用全部基因）
    parser.add_argument('--n_hvg', default='all', help="高变基因数量，或'all'使用全部基因（默认'all'）")
    # 新增：从JSON加载配置（支持meta风格或扁平字典）
    parser.add_argument('--config_json', default=None, help='从JSON文件加载超参数配置（支持 {"cfg": {...}} 或直接扁平字典）')
    
    # 新增：网络结构与优化器超参数
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--lr_scheduler', choices=['none', 'linear', 'cosine', 'warmup_cosine', 'plateau'], default=None, help='学习率调度模式')
    parser.add_argument('--lr_warmup_epochs', type=int, default=None, help='warmup_cosine 的 warmup epoch 数')
    parser.add_argument('--min_lr_ratio', type=float, default=None, help='最小学习率比例，min_lr = lr * min_lr_ratio')
    parser.add_argument('--plateau_metric', choices=['val_accuracy', 'val_avg_total_loss', 'val_macro_f1', 'val_auroc', 'val_auprc'], default=None, help='plateau 模式监控的验证指标')
    parser.add_argument('--plateau_factor', type=float, default=None, help='plateau 模式学习率衰减倍率')
    parser.add_argument('--plateau_patience', type=int, default=None, help='plateau 模式耐心值')
    parser.add_argument('--plateau_threshold', type=float, default=None, help='plateau 模式指标改进阈值')
    parser.add_argument('--plateau_cooldown', type=int, default=None, help='plateau 模式 cooldown epoch 数')
    parser.add_argument('--hidden', type=int, default=None, help='旧版兼容参数：统一隐藏层维度；不要与 --GNN_hidden 同时传入')
    parser.add_argument('--GNN_hidden', default=None, help='GNN每层隐藏维度；支持单个整数或逗号分隔列表，默认：256,128,64')
    parser.add_argument('--num_layers', type=int, default=None, help='GNN层数')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout比例')
    # 新增：分类头与域头结构
    parser.add_argument('--clf_hidden', default=None, help='分类头隐藏层维度，逗号分隔的正整数列表（默认：256,128,64）')
    parser.add_argument('--dom_hidden', type=int, default=None, help='域头隐藏层维度（默认：64）')
    
    # 新增：设备控制
    parser.add_argument('--device', default=None, help='指定设备（cpu/cuda）,不指定则自动检测')
    
    # 新增：双域对抗控制与权重（新参数优先级最高）
    parser.add_argument('--use_domain_adv_slide', type=int, choices=[0,1], default=None, help='启用/关闭批次域对抗（1/0）')
    parser.add_argument('--use_domain_adv_cancer', type=int, choices=[0,1], default=None, help='启用/关闭癌种域对抗（1/0）')
    parser.add_argument('--lambda_slide', type=float, default=None, help='批次域对抗损失权重')
    parser.add_argument('--lambda_cancer', type=float, default=None, help='癌种域对抗损失权重')
    # 新增：GRL beta schedule（默认 DANN-style；可切换 constant / linear）
    parser.add_argument('--grl_beta_mode', choices=['dann', 'constant', 'linear'], default=None, help='GRL beta 模式：dann(支持delay后在剩余训练过程上走完整DANN曲线)/constant(全程恒定=target)/linear(delay后线性升到target)')
    parser.add_argument('--grl_beta_slide_target', type=float, default=None, help='批次域 GRL 目标强度，默认1.0')
    parser.add_argument('--grl_beta_cancer_target', type=float, default=None, help='癌种域 GRL 目标强度，默认0.5')
    parser.add_argument('--grl_beta_gamma', type=float, default=None, help='GRL DANN schedule gamma，默认10；仅在 grl_beta_mode=dann 时生效')
    parser.add_argument('--grl_beta_slide_delay_epochs', type=int, default=None, help='批次域 GRL delay epoch 数；对 dann 和 linear 生效，默认1')
    parser.add_argument('--grl_beta_slide_warmup_epochs', type=int, default=None, help='批次域 GRL 线性 warm-up epoch 数；仅对 linear 生效，默认8')
    parser.add_argument('--grl_beta_cancer_delay_epochs', type=int, default=None, help='癌种域 GRL delay epoch 数；对 dann 和 linear 生效，默认3')
    parser.add_argument('--grl_beta_cancer_warmup_epochs', type=int, default=None, help='癌种域 GRL 线性 warm-up epoch 数；仅对 linear 生效，默认12')
    # 新增：MMD 对齐
    parser.add_argument('--use_mmd', type=int, choices=[0,1], default=None, help='启用/关闭 MMD 对齐（1/0）')
    parser.add_argument('--mmd_on', choices=['slide', 'cancer', 'both'], default=None, help='MMD 对齐域：slide/cancer/both')
    parser.add_argument('--lambda_mmd', type=float, default=None, help='MMD 损失权重')
    parser.add_argument('--mmd_num_kernels', type=int, default=None, help='MMD 多带宽 RBF 的核个数')
    parser.add_argument('--mmd_kernel_mul', type=float, default=None, help='MMD 多带宽倍率')
    parser.add_argument('--mmd_sigma', type=float, default=None, help='固定 RBF 带宽；不传则自适应估计')
    parser.add_argument('--mmd_max_pairs', type=int, default=None, help='每个 batch 最多计算多少个域对')
    parser.add_argument('--mmd_spots_per_slide', type=int, default=None, help='每张切片最多抽取多少个 spot 参与 MMD；<=0 表示使用全部 spot')

    # 新增：generated-support Wasserstein barycenter 对齐
    parser.add_argument('--use_wb_align', type=int, choices=[0, 1], default=None, help='启用/关闭 generated-support WB 对齐（1/0）')
    parser.add_argument('--wb_loss_type', choices=['dual_potential', 'euclidean_pairwise'], default=None, help='WB loss 类型')
    parser.add_argument('--wb_support_mode', choices=['generated_support'], default=None, help='WB support 生成方式；第一版仅支持 generated_support')
    parser.add_argument('--lambda_wb', type=float, default=None, help='WB loss 最大权重')
    parser.add_argument('--wb_warmup_epochs', type=int, default=None, help='WB loss 开始前 warmup epoch 数')
    parser.add_argument('--wb_ramp_epochs', type=int, default=None, help='WB loss 从0 ramp到lambda_wb的 epoch 数')
    parser.add_argument('--wb_support_hidden', type=int, default=None, help='GeneratedSupportMap hidden 维度')
    parser.add_argument('--wb_support_dropout', type=float, default=None, help='GeneratedSupportMap dropout')
    parser.add_argument('--wb_anchor_weight', type=float, default=None, help='h-b anchor loss 在 WB 模块内的相对权重')
    parser.add_argument('--wb_potential_hidden', type=int, default=None, help='WB potential MLP hidden 维度')
    parser.add_argument('--wb_potential_lr', type=float, default=None, help='WB potential optimizer 学习率')
    parser.add_argument('--wb_pot_every_n_steps', type=int, default=None, help='每多少个 main step 更新一次 potentials')
    parser.add_argument('--wb_spots_per_graph', type=int, default=None, help='每张 graph/slide 最多抽取多少 spot 参与 WB；<=0 表示不限制')
    parser.add_argument('--wb_spots_per_cancer', type=int, default=None, help='每个 cancer 最多抽取多少 spot 参与 WB；<=0 表示关闭二级cap')
    parser.add_argument('--wb_support_size', type=int, default=None, help='dual_potential support-side b 子采样数量；<=0 表示使用全部 selected b')
    parser.add_argument('--wb_min_cancers', type=int, default=None, help='计算 WB loss 所需最少 active cancer 数')
    parser.add_argument('--wb_min_spots', type=int, default=None, help='每个 active cancer 所需最少 spot 数')
    parser.add_argument('--wb_regularizer', choices=['entropy', 'l2'], default=None, help='dual_potential regularizer')
    parser.add_argument('--wb_epsilon', type=float, default=None, help='dual_potential regularizer epsilon')
    parser.add_argument('--wb_label_balanced_sampling', type=int, choices=[0, 1], default=None, help='WB 内部是否启用 label-balanced spot sampling')
    parser.add_argument('--wb_state_direction', type=int, choices=[0, 1], default=None, help='是否启用 tumor-normal shared state direction 约束')
    parser.add_argument('--wb_state_direction_weight', type=float, default=None, help='state direction loss 在 raw WB loss 中的权重')
    parser.add_argument('--wb_eval_loss', type=int, choices=[0, 1], default=None, help='验证阶段是否计算 WB diagnostics/loss')
    parser.add_argument('--wb_euclid_pairwise_weight', type=float, default=None, help='euclidean_pairwise 分布项权重')
    parser.add_argument('--wb_potential_weight', type=float, default=None, help='euclidean_pairwise potential main 项权重')
    parser.add_argument('--wb_potential_constraint_weight', type=float, default=None, help='potential update 零均值约束权重')
    parser.add_argument('--best_metric', choices=['val_macro_f1', 'val_auprc', 'val_accuracy', 'val_auroc', 'val_avg_total_loss'], default=None, help='最佳 checkpoint 选择指标')
    
    # 新增：癌种分层与K折/LOCO
    parser.add_argument('--stratify_by_cancer', action='store_true', default=True, help='启用癌种分层划分：按比例分配验证集且每癌种保底1张（n=1除外）')
    parser.add_argument('--no_stratify_by_cancer', action='store_false', dest='stratify_by_cancer', help='关闭癌种分层，使用简单划分（最后1张为验证）')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例（默认0.2，按癌种分配且保底1张）')
    parser.add_argument('--kfold_cancer', type=int, default=None, help='整数输入，基于癌种的K折（按比例划分验证集并随机组合）。产物统一保存至artifacts_dir同级目录下，并在/kfold_val/kfold_summary.csv 汇总指标；这里的artifacts_dir 仅决定保存根目录。')
    parser.add_argument('--split_seed', type=int, default=42, help='分层/交叉验证随机种子')
    parser.add_argument('--split_test_only', action='store_true', help='仅测试划分逻辑，不进行训练，打印每折的统计信息')
    parser.add_argument('--leave_one_cancer_out', action='store_true', help='启用留一癌种评估模式（对每个癌种单独训练：其为验证，其余为训练）')
    parser.add_argument('--loco_resume', action='store_true', help='LOCO模式断点续跑：若 loco_eval/{CancerType}/meta.json 已存在且 cfg 匹配，则跳过该癌种训练')
    parser.add_argument('--loco_force_rerun', action='store_true', help='LOCO模式强制重跑：即使已存在结果也重新训练并覆盖（优先级高于 --loco_resume）')
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
    # 新增：保存最后一个epoch（仅当实际跑满 epochs 时覆盖 model.pt，同时保存 best 到 model_best.pt）
    parser.add_argument('--save_last', action='store_true',
                        help='若训练实际跑满 epochs：用最后一个epoch覆盖 artifacts_dir/model.pt；并额外保存最优模型到 artifacts_dir/model_best.pt（不强制关闭早停）')
    parser.add_argument('--save_epoch_checkpoints', type=int, nargs='*', default=None,
                        help='额外保存指定epoch的模型快照，例如 --save_epoch_checkpoints 100 200；输出到 artifacts_dir/epoch_checkpoints/epoch_XXX/')

    args = parser.parse_args()

    cfg = {'pca_dim':64, 'lap_pe_dim':0, 'knn_k':6, 'gaussian_sigma_factor':1.0, 'num_layers':3, 'dropout':0.3, 'model':'gatv2', 'heads':4, 'lr':1e-3, 'weight_decay':1e-4, 'lr_scheduler':'none', 'lr_warmup_epochs':10, 'min_lr_ratio':0.01, 'plateau_metric':'val_accuracy', 'plateau_factor':0.5, 'plateau_patience':10, 'plateau_threshold':1e-4, 'plateau_cooldown':0, 'epochs':100, 'batch_size_graphs':2, 'early_patience':30,
           'sampler_mode': 'random',
           'sampler_k_cancers': None,
           'sampler_m_per_cancer': None,
           'sampler_enforce_distinct_batch': True,
           'sampler_seed': None,
           'subgraph_mode': 'off',
           'subgraph_target_spots': 1000,
           'subgraph_min_spots': 300,
           # 控制项
           'use_pca': False,
           # 方案1：早期融合（默认关闭以保持旧行为）
           'use_image_features': False,
           'img_use_pca': True,
           'img_pca_dim': 256,
           'concat_lap_pe': True,
           'lap_pe_use_gaussian': False,
           # 分类头/域头
           'clf_hidden': [256, 128, 64],
           'clf_latent_dim': 64,
           'dom_hidden': 64,
           # 新增：双域默认配置（新字段）
           'use_domain_adv_slide': True,   # 默认开启（batch/slide 域）
           'use_domain_adv_cancer': True, # 默认开启
           # alpha（域 loss 权重）
	           'lambda_slide': 1.0,
	           'lambda_cancer': 1.0,
	           # beta（GRL 对抗强度）
	           'grl_beta_mode': 'dann',
	           'grl_beta_slide_target': 1.0,
	           'grl_beta_cancer_target': 0.5,
	           'grl_beta_gamma': 10.0,
               'grl_beta_slide_delay_epochs': 1,
               'grl_beta_slide_warmup_epochs': 8,
               'grl_beta_cancer_delay_epochs': 3,
               'grl_beta_cancer_warmup_epochs': 12,
               # 新增：MMD 默认配置
               'use_mmd': False,
               'mmd_on': 'slide',
               'lambda_mmd': 0.05,
               'mmd_num_kernels': 5,
               'mmd_kernel_mul': 2.0,
               'mmd_sigma': None,
               'mmd_max_pairs': 8,
               'mmd_spots_per_slide': 0,
               # 新增：generated-support Wasserstein barycenter 默认配置
               'use_wb_align': False,
               'wb_loss_type': 'euclidean_pairwise',
               'wb_support_mode': 'generated_support',
               'lambda_wb': 0.01,
               'wb_warmup_epochs': 10,
               'wb_ramp_epochs': 20,
               'wb_support_hidden': 128,
               'wb_support_dropout': 0.0,
               'wb_anchor_weight': 0.5,
               'wb_potential_hidden': 128,
               'wb_potential_lr': 1e-4,
               'wb_pot_every_n_steps': 1,
               'wb_spots_per_graph': 64,
               'wb_spots_per_cancer': 0,
               'wb_support_size': 128,
               'wb_min_cancers': 2,
               'wb_min_spots': 2,
               'wb_regularizer': 'l2',
               'wb_epsilon': 0.1,
               'wb_label_balanced_sampling': False,
               'wb_state_direction': False,
               'wb_state_direction_weight': 0.1,
               'wb_eval_loss': False,
               'wb_euclid_pairwise_weight': 1.0,
               'wb_potential_weight': 1.0,
               'wb_potential_constraint_weight': 0.01,
               'best_metric': 'val_macro_f1',
	           # 新增：HVG控制
	           'n_hvg': 'all'
	           }

    # 先从JSON加载作为基准配置（若提供）
    cfg_json_keys = set()
    if args.config_json is not None:
        try:
            with open(args.config_json, 'r', encoding='utf-8') as f:
                cfg_json = json.load(f)
            if isinstance(cfg_json, dict):
                if 'cfg' in cfg_json and isinstance(cfg_json['cfg'], dict):
                    cfg_json_keys = set(cfg_json['cfg'].keys())
                    cfg.update(cfg_json['cfg'])
                else:
                    cfg_json_keys = set(cfg_json.keys())
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
    if getattr(args, 'sampler_mode', None) is not None:
        cfg['sampler_mode'] = str(args.sampler_mode)
    if getattr(args, 'sampler_k_cancers', None) is not None:
        cfg['sampler_k_cancers'] = int(args.sampler_k_cancers)
    if getattr(args, 'sampler_m_per_cancer', None) is not None:
        cfg['sampler_m_per_cancer'] = int(args.sampler_m_per_cancer)
    if getattr(args, 'sampler_enforce_distinct_batch', None) is not None:
        cfg['sampler_enforce_distinct_batch'] = bool(args.sampler_enforce_distinct_batch)
    if getattr(args, 'subgraph_mode', None) is not None:
        cfg['subgraph_mode'] = str(args.subgraph_mode)
    if getattr(args, 'subgraph_target_spots', None) is not None:
        cfg['subgraph_target_spots'] = int(args.subgraph_target_spots)
    if getattr(args, 'subgraph_min_spots', None) is not None:
        cfg['subgraph_min_spots'] = int(args.subgraph_min_spots)
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
    if getattr(args, 'use_image_features', None) is not None:
        cfg['use_image_features'] = bool(args.use_image_features)
    if getattr(args, 'img_use_pca', None) is not None:
        cfg['img_use_pca'] = bool(args.img_use_pca)
    if getattr(args, 'img_pca_dim', None) is not None:
        cfg['img_pca_dim'] = int(args.img_pca_dim)
    # 新增：覆盖 HVG 数量（字符串'all'或可解析为整数的字符串/整数）
    if getattr(args, 'n_hvg', None) is not None:
        cfg['n_hvg'] = args.n_hvg
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.weight_decay is not None:
        cfg['weight_decay'] = args.weight_decay
    if getattr(args, 'lr_scheduler', None) is not None:
        cfg['lr_scheduler'] = str(args.lr_scheduler)
    if getattr(args, 'lr_warmup_epochs', None) is not None:
        cfg['lr_warmup_epochs'] = int(args.lr_warmup_epochs)
    if getattr(args, 'min_lr_ratio', None) is not None:
        cfg['min_lr_ratio'] = float(args.min_lr_ratio)
    if getattr(args, 'plateau_metric', None) is not None:
        cfg['plateau_metric'] = str(args.plateau_metric)
    if getattr(args, 'plateau_factor', None) is not None:
        cfg['plateau_factor'] = float(args.plateau_factor)
    if getattr(args, 'plateau_patience', None) is not None:
        cfg['plateau_patience'] = int(args.plateau_patience)
    if getattr(args, 'plateau_threshold', None) is not None:
        cfg['plateau_threshold'] = float(args.plateau_threshold)
    if getattr(args, 'plateau_cooldown', None) is not None:
        cfg['plateau_cooldown'] = int(args.plateau_cooldown)
    if args.hidden is not None:
        cfg['hidden'] = args.hidden
    if args.GNN_hidden is not None:
        cfg['GNN_hidden'] = args.GNN_hidden
    if args.num_layers is not None:
        cfg['num_layers'] = args.num_layers
    if args.dropout is not None:
        cfg['dropout'] = args.dropout
    if getattr(args, 'clf_hidden', None) is not None:
        # 逗号分隔，例如 "256,128,64"
        s = str(args.clf_hidden).strip()
        if s:
            dims = [int(x.strip()) for x in s.split(',') if x.strip() != '']
            if not dims:
                raise ValueError('--clf_hidden must contain at least 1 integer')
            if any(int(d) <= 0 for d in dims):
                raise ValueError(f'--clf_hidden must contain only positive integers, got: {dims}')
            cfg['clf_hidden'] = [int(d) for d in dims]
    if getattr(args, 'dom_hidden', None) is not None:
        cfg['dom_hidden'] = int(args.dom_hidden)

    # 兼容：若从 config_json 读入 clf_hidden/dom_hidden，做一次校验与规范化
    if 'clf_hidden' in cfg:
        dims = cfg['clf_hidden']
        if isinstance(dims, str):
            dims = [int(x.strip()) for x in dims.split(',') if x.strip() != '']
        if not isinstance(dims, (list, tuple)) or len(dims) < 1:
            raise ValueError(f"cfg['clf_hidden'] must be a non-empty list/tuple of positive ints, got: {dims}")
        dims = [int(x) for x in dims]
        if any(int(x) <= 0 for x in dims):
            raise ValueError(f"cfg['clf_hidden'] must contain only positive ints, got: {dims}")
        cfg['clf_hidden'] = dims
        cfg['clf_latent_dim'] = int(dims[-1])
    else:
        cfg['clf_hidden'] = [256, 128, 64]
        cfg['clf_latent_dim'] = 64
    if 'dom_hidden' in cfg:
        cfg['dom_hidden'] = int(cfg['dom_hidden'])

    # 新增：双域对抗参数（新命令行优先级最高）
    if args.use_domain_adv_slide is not None:
        cfg['use_domain_adv_slide'] = bool(args.use_domain_adv_slide)
    if args.use_domain_adv_cancer is not None:
        cfg['use_domain_adv_cancer'] = bool(args.use_domain_adv_cancer)
    if args.lambda_slide is not None:
        cfg['lambda_slide'] = float(args.lambda_slide)
    if args.lambda_cancer is not None:
        cfg['lambda_cancer'] = float(args.lambda_cancer)
    if getattr(args, 'grl_beta_mode', None) is not None:
        cfg['grl_beta_mode'] = str(args.grl_beta_mode)
    if getattr(args, 'grl_beta_slide_target', None) is not None:
        cfg['grl_beta_slide_target'] = float(args.grl_beta_slide_target)
    if getattr(args, 'grl_beta_cancer_target', None) is not None:
        cfg['grl_beta_cancer_target'] = float(args.grl_beta_cancer_target)
    if getattr(args, 'grl_beta_gamma', None) is not None:
        cfg['grl_beta_gamma'] = float(args.grl_beta_gamma)
    if getattr(args, 'grl_beta_slide_delay_epochs', None) is not None:
        cfg['grl_beta_slide_delay_epochs'] = int(args.grl_beta_slide_delay_epochs)
    if getattr(args, 'grl_beta_slide_warmup_epochs', None) is not None:
        cfg['grl_beta_slide_warmup_epochs'] = int(args.grl_beta_slide_warmup_epochs)
    if getattr(args, 'grl_beta_cancer_delay_epochs', None) is not None:
        cfg['grl_beta_cancer_delay_epochs'] = int(args.grl_beta_cancer_delay_epochs)
    if getattr(args, 'grl_beta_cancer_warmup_epochs', None) is not None:
        cfg['grl_beta_cancer_warmup_epochs'] = int(args.grl_beta_cancer_warmup_epochs)
    if getattr(args, 'use_mmd', None) is not None:
        cfg['use_mmd'] = bool(args.use_mmd)
    if getattr(args, 'mmd_on', None) is not None:
        cfg['mmd_on'] = str(args.mmd_on)
    if getattr(args, 'lambda_mmd', None) is not None:
        cfg['lambda_mmd'] = float(args.lambda_mmd)
    if getattr(args, 'mmd_num_kernels', None) is not None:
        cfg['mmd_num_kernels'] = int(args.mmd_num_kernels)
    if getattr(args, 'mmd_kernel_mul', None) is not None:
        cfg['mmd_kernel_mul'] = float(args.mmd_kernel_mul)
    if getattr(args, 'mmd_sigma', None) is not None:
        cfg['mmd_sigma'] = float(args.mmd_sigma)
    if getattr(args, 'mmd_max_pairs', None) is not None:
        cfg['mmd_max_pairs'] = int(args.mmd_max_pairs)
    if getattr(args, 'mmd_spots_per_slide', None) is not None:
        cfg['mmd_spots_per_slide'] = int(args.mmd_spots_per_slide)
    if getattr(args, 'use_wb_align', None) is not None:
        cfg['use_wb_align'] = bool(args.use_wb_align)
    if getattr(args, 'wb_loss_type', None) is not None:
        cfg['wb_loss_type'] = str(args.wb_loss_type)
    if getattr(args, 'wb_support_mode', None) is not None:
        cfg['wb_support_mode'] = str(args.wb_support_mode)
    if getattr(args, 'lambda_wb', None) is not None:
        cfg['lambda_wb'] = float(args.lambda_wb)
    if getattr(args, 'wb_warmup_epochs', None) is not None:
        cfg['wb_warmup_epochs'] = int(args.wb_warmup_epochs)
    if getattr(args, 'wb_ramp_epochs', None) is not None:
        cfg['wb_ramp_epochs'] = int(args.wb_ramp_epochs)
    if getattr(args, 'wb_support_hidden', None) is not None:
        cfg['wb_support_hidden'] = int(args.wb_support_hidden)
    if getattr(args, 'wb_support_dropout', None) is not None:
        cfg['wb_support_dropout'] = float(args.wb_support_dropout)
    if getattr(args, 'wb_anchor_weight', None) is not None:
        cfg['wb_anchor_weight'] = float(args.wb_anchor_weight)
    if getattr(args, 'wb_potential_hidden', None) is not None:
        cfg['wb_potential_hidden'] = int(args.wb_potential_hidden)
    if getattr(args, 'wb_potential_lr', None) is not None:
        cfg['wb_potential_lr'] = float(args.wb_potential_lr)
    if getattr(args, 'wb_pot_every_n_steps', None) is not None:
        cfg['wb_pot_every_n_steps'] = int(args.wb_pot_every_n_steps)
    if getattr(args, 'wb_spots_per_graph', None) is not None:
        cfg['wb_spots_per_graph'] = int(args.wb_spots_per_graph)
    if getattr(args, 'wb_spots_per_cancer', None) is not None:
        cfg['wb_spots_per_cancer'] = int(args.wb_spots_per_cancer)
    if getattr(args, 'wb_support_size', None) is not None:
        cfg['wb_support_size'] = int(args.wb_support_size)
    if getattr(args, 'wb_min_cancers', None) is not None:
        cfg['wb_min_cancers'] = int(args.wb_min_cancers)
    if getattr(args, 'wb_min_spots', None) is not None:
        cfg['wb_min_spots'] = int(args.wb_min_spots)
    if getattr(args, 'wb_regularizer', None) is not None:
        cfg['wb_regularizer'] = str(args.wb_regularizer)
    if getattr(args, 'wb_epsilon', None) is not None:
        cfg['wb_epsilon'] = float(args.wb_epsilon)
    if getattr(args, 'wb_label_balanced_sampling', None) is not None:
        cfg['wb_label_balanced_sampling'] = bool(args.wb_label_balanced_sampling)
    if getattr(args, 'wb_state_direction', None) is not None:
        cfg['wb_state_direction'] = bool(args.wb_state_direction)
    if getattr(args, 'wb_state_direction_weight', None) is not None:
        cfg['wb_state_direction_weight'] = float(args.wb_state_direction_weight)
    if getattr(args, 'wb_eval_loss', None) is not None:
        cfg['wb_eval_loss'] = bool(args.wb_eval_loss)
    if getattr(args, 'wb_euclid_pairwise_weight', None) is not None:
        cfg['wb_euclid_pairwise_weight'] = float(args.wb_euclid_pairwise_weight)
    if getattr(args, 'wb_potential_weight', None) is not None:
        cfg['wb_potential_weight'] = float(args.wb_potential_weight)
    if getattr(args, 'wb_potential_constraint_weight', None) is not None:
        cfg['wb_potential_constraint_weight'] = float(args.wb_potential_constraint_weight)
    if getattr(args, 'best_metric', None) is not None:
        cfg['best_metric'] = str(args.best_metric)

    # 默认值填充（新字段）
    if cfg.get('use_domain_adv_slide', None) is None:
        cfg['use_domain_adv_slide'] = True
    cfg['use_domain_adv_slide'] = bool(cfg['use_domain_adv_slide'])
    cfg['use_domain_adv_cancer'] = bool(cfg.get('use_domain_adv_cancer', True))
    if cfg.get('lambda_slide', None) is None:
        cfg['lambda_slide'] = 1.0
    if cfg.get('lambda_cancer', None) is None:
        cfg['lambda_cancer'] = 1.0
    if cfg.get('grl_beta_mode', None) is None:
        cfg['grl_beta_mode'] = 'dann'
    if cfg.get('grl_beta_slide_target', None) is None:
        cfg['grl_beta_slide_target'] = 1.0
    if cfg.get('grl_beta_cancer_target', None) is None:
        cfg['grl_beta_cancer_target'] = 0.5
    if cfg.get('grl_beta_gamma', None) is None:
        cfg['grl_beta_gamma'] = 10.0
    if cfg.get('grl_beta_slide_delay_epochs', None) is None:
        cfg['grl_beta_slide_delay_epochs'] = 1
    if cfg.get('grl_beta_slide_warmup_epochs', None) is None:
        cfg['grl_beta_slide_warmup_epochs'] = 8
    if cfg.get('grl_beta_cancer_delay_epochs', None) is None:
        cfg['grl_beta_cancer_delay_epochs'] = 3
    if cfg.get('grl_beta_cancer_warmup_epochs', None) is None:
        cfg['grl_beta_cancer_warmup_epochs'] = 12
    if str(cfg.get('grl_beta_mode', 'dann')) not in {'dann', 'constant', 'linear'}:
        raise ValueError(f"cfg['grl_beta_mode'] must be 'dann', 'constant' or 'linear', got: {cfg.get('grl_beta_mode')}")
    for key in (
        'grl_beta_slide_delay_epochs',
        'grl_beta_slide_warmup_epochs',
        'grl_beta_cancer_delay_epochs',
        'grl_beta_cancer_warmup_epochs',
    ):
        value = cfg.get(key, 0)
        if isinstance(value, bool):
            raise ValueError(f"cfg['{key}'] must be an integer epoch count, got bool: {value}")
        if isinstance(value, float) and not float(value).is_integer():
            raise ValueError(f"cfg['{key}'] must be an integer epoch count, got: {value}")
        cfg[key] = int(value)
    cfg['use_mmd'] = bool(cfg.get('use_mmd', False))
    cfg['mmd_on'] = str(cfg.get('mmd_on', 'slide')).lower()
    if cfg['mmd_on'] not in {'slide', 'cancer', 'both'}:
        raise ValueError(f"cfg['mmd_on'] must be one of slide/cancer/both, got: {cfg['mmd_on']}")
    cfg['lambda_mmd'] = float(cfg.get('lambda_mmd', 0.05))
    cfg['mmd_num_kernels'] = int(cfg.get('mmd_num_kernels', 5))
    cfg['mmd_kernel_mul'] = float(cfg.get('mmd_kernel_mul', 2.0))
    cfg['mmd_sigma'] = None if cfg.get('mmd_sigma', None) is None else float(cfg['mmd_sigma'])
    cfg['mmd_max_pairs'] = int(cfg.get('mmd_max_pairs', 8))
    cfg['mmd_spots_per_slide'] = int(cfg.get('mmd_spots_per_slide', 0))
    cfg['use_wb_align'] = bool(cfg.get('use_wb_align', False))
    cfg['wb_loss_type'] = str(cfg.get('wb_loss_type', 'euclidean_pairwise')).lower()
    if cfg['wb_loss_type'] not in {'dual_potential', 'euclidean_pairwise'}:
        raise ValueError(f"cfg['wb_loss_type'] must be dual_potential or euclidean_pairwise, got: {cfg['wb_loss_type']}")
    cfg['wb_support_mode'] = str(cfg.get('wb_support_mode', 'generated_support')).lower()
    if cfg['wb_support_mode'] != 'generated_support':
        raise ValueError("First WB implementation only supports cfg['wb_support_mode']='generated_support'")
    cfg['lambda_wb'] = float(cfg.get('lambda_wb', 0.01))
    cfg['wb_warmup_epochs'] = int(cfg.get('wb_warmup_epochs', 10))
    cfg['wb_ramp_epochs'] = int(cfg.get('wb_ramp_epochs', 20))
    cfg['wb_support_hidden'] = int(cfg.get('wb_support_hidden', 128))
    cfg['wb_support_dropout'] = float(cfg.get('wb_support_dropout', 0.0))
    cfg['wb_anchor_weight'] = float(cfg.get('wb_anchor_weight', 0.5))
    cfg['wb_potential_hidden'] = int(cfg.get('wb_potential_hidden', 128))
    cfg['wb_potential_lr'] = float(cfg.get('wb_potential_lr', 1e-4))
    cfg['wb_pot_every_n_steps'] = max(1, int(cfg.get('wb_pot_every_n_steps', 1)))
    cfg['wb_spots_per_graph'] = int(cfg.get('wb_spots_per_graph', 64))
    cfg['wb_spots_per_cancer'] = int(cfg.get('wb_spots_per_cancer', 0))
    cfg['wb_support_size'] = int(cfg.get('wb_support_size', 128))
    cfg['wb_min_cancers'] = max(1, int(cfg.get('wb_min_cancers', 2)))
    cfg['wb_min_spots'] = max(1, int(cfg.get('wb_min_spots', 2)))
    cfg['wb_regularizer'] = str(cfg.get('wb_regularizer', 'l2')).lower()
    if cfg['wb_regularizer'] not in {'l2', 'entropy'}:
        raise ValueError(f"cfg['wb_regularizer'] must be l2 or entropy, got: {cfg['wb_regularizer']}")
    cfg['wb_epsilon'] = float(cfg.get('wb_epsilon', 0.1))
    if cfg['wb_epsilon'] <= 0:
        raise ValueError(f"cfg['wb_epsilon'] must be > 0, got: {cfg['wb_epsilon']}")
    cfg['wb_label_balanced_sampling'] = bool(cfg.get('wb_label_balanced_sampling', False))
    cfg['wb_state_direction'] = bool(cfg.get('wb_state_direction', False))
    cfg['wb_state_direction_weight'] = float(cfg.get('wb_state_direction_weight', 0.1))
    cfg['wb_eval_loss'] = bool(cfg.get('wb_eval_loss', False))
    cfg['wb_euclid_pairwise_weight'] = float(cfg.get('wb_euclid_pairwise_weight', 1.0))
    cfg['wb_potential_weight'] = float(cfg.get('wb_potential_weight', 1.0))
    cfg['wb_potential_constraint_weight'] = float(cfg.get('wb_potential_constraint_weight', 0.01))
    cfg['best_metric'] = str(cfg.get('best_metric', 'val_macro_f1')).lower()
    if cfg['best_metric'] not in {'val_macro_f1', 'val_auprc', 'val_accuracy', 'val_auroc', 'val_avg_total_loss'}:
        raise ValueError(
            "cfg['best_metric'] must be one of val_macro_f1/val_auprc/val_accuracy/"
            f"val_auroc/val_avg_total_loss, got: {cfg['best_metric']}"
        )
    if cfg['use_wb_align'] and cfg['use_mmd']:
        print("[WB] Warning: use_wb_align=1 and use_mmd=1 are both enabled. WB is intended to replace/upgrade MMD; continuing because both were requested.")
    if cfg['use_wb_align'] and bool(cfg.get('use_domain_adv_cancer', False)):
        user_set_lambda_cancer = (getattr(args, 'lambda_cancer', None) is not None) or ('lambda_cancer' in cfg_json_keys)
        if not user_set_lambda_cancer:
            cfg['lambda_cancer'] = 0.1
            print("[WB] use_wb_align=1: lambda_cancer was not explicitly set; using weak cancer GRL lambda_cancer=0.1.")
        elif float(cfg.get('lambda_cancer', 1.0)) > 0.1:
            print(
                f"[WB] Warning: lambda_cancer={cfg.get('lambda_cancer')} with WB may over-align cancer information; "
                "recommended WB experiments use 0.05-0.1."
            )
    if cfg['use_wb_align'] and str(cfg.get('sampler_mode', 'random')).lower() != 'cancer_balanced':
        print(
            "[WB] Warning: generated-support WB benefits from multi-cancer batches. "
            "Recommended: --batch_size_graphs 8 --sampler_mode cancer_balanced --sampler_k_cancers 4 --sampler_m_per_cancer 2."
        )
    if cfg.get('sampler_seed', None) is None:
        cfg['sampler_seed'] = int(getattr(args, 'split_seed', 42))
    cfg['lr_scheduler'] = str(cfg.get('lr_scheduler', 'none')).lower()
    if cfg['lr_scheduler'] not in {'none', 'linear', 'cosine', 'warmup_cosine', 'plateau'}:
        raise ValueError(f"cfg['lr_scheduler'] must be one of none/linear/cosine/warmup_cosine/plateau, got: {cfg['lr_scheduler']}")
    if float(cfg.get('lr', 0.0)) <= 0:
        raise ValueError(f"cfg['lr'] must be > 0, got: {cfg.get('lr')}")
    cfg['min_lr_ratio'] = float(cfg.get('min_lr_ratio', 0.01))
    if not (0.0 < cfg['min_lr_ratio'] <= 1.0):
        raise ValueError(f"cfg['min_lr_ratio'] must be in (0, 1], got: {cfg['min_lr_ratio']}")
    for key, default_value in (('lr_warmup_epochs', 10), ('plateau_patience', 10), ('plateau_cooldown', 0)):
        value = cfg.get(key, default_value)
        if value is None:
            value = default_value
        if isinstance(value, bool):
            raise ValueError(f"cfg['{key}'] must be an integer, got bool: {value}")
        if isinstance(value, float) and not float(value).is_integer():
            raise ValueError(f"cfg['{key}'] must be an integer, got: {value}")
        cfg[key] = int(value)
    cfg['plateau_metric'] = str(cfg.get('plateau_metric', 'val_accuracy'))
    if cfg['plateau_metric'] not in {'val_accuracy', 'val_avg_total_loss', 'val_macro_f1', 'val_auroc', 'val_auprc'}:
        raise ValueError(f"cfg['plateau_metric'] must be one of val_accuracy/val_avg_total_loss/val_macro_f1/val_auroc/val_auprc, got: {cfg['plateau_metric']}")
    cfg['plateau_factor'] = float(cfg.get('plateau_factor', 0.5))
    if not (0.0 < cfg['plateau_factor'] < 1.0):
        raise ValueError(f"cfg['plateau_factor'] must be in (0, 1), got: {cfg['plateau_factor']}")
    cfg['plateau_threshold'] = float(cfg.get('plateau_threshold', 1e-4))
    if cfg['plateau_threshold'] < 0:
        raise ValueError(f"cfg['plateau_threshold'] must be >= 0, got: {cfg['plateau_threshold']}")
    if cfg['plateau_patience'] < 0:
        raise ValueError(f"cfg['plateau_patience'] must be >= 0, got: {cfg['plateau_patience']}")
    if cfg['plateau_cooldown'] < 0:
        raise ValueError(f"cfg['plateau_cooldown'] must be >= 0, got: {cfg['plateau_cooldown']}")
    cfg = normalize_gnn_config(cfg)
    cfg = normalize_sampler_config(cfg)
    args._save_epoch_checkpoints = _normalize_save_epoch_checkpoints(getattr(args, 'save_epoch_checkpoints', None))
    invalid_save_epochs = [ep for ep in args._save_epoch_checkpoints if int(ep) > int(cfg.get('epochs', 0))]
    if len(invalid_save_epochs) > 0:
        raise ValueError(
            f"save_epoch_checkpoints contains epochs beyond cfg['epochs']={cfg.get('epochs')}: {invalid_save_epochs}"
        )

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
    files = set(data.files)
    Xs = list(data['Xs']); ys = list(data['ys']); xys = list(data['xys']); slide_ids = list(data['slide_ids'])
    gene_names = list(data['gene_names'])

    use_image_features = bool(cfg.get('use_image_features', False))
    X_imgs = None
    img_masks = None
    img_feature_names = None
    if use_image_features:
        required = {'X_imgs', 'img_masks', 'img_feature_names'}
        missing = sorted(list(required - files))
        if missing:
            raise ValueError(f"use_image_features=1 requires keys {sorted(required)} in train_npz, missing: {missing}")
        X_imgs = list(data['X_imgs'])
        img_masks = list(data['img_masks'])
        img_feature_names = list(data['img_feature_names'])
        if not (len(X_imgs) == len(img_masks) == len(Xs)):
            raise ValueError(
                f'Length mismatch in train_npz: len(Xs)={len(Xs)}, len(X_imgs)={len(X_imgs)}, len(img_masks)={len(img_masks)}'
            )

    slides = []
    for i, (X, y, xy, sid) in enumerate(zip(Xs, ys, xys, slide_ids)):
        item = {'X': X, 'y': y, 'xy': xy, 'slide_id': sid}
        if use_image_features:
            item['X_img'] = X_imgs[i]
            item['img_mask'] = img_masks[i]
        slides.append(item)

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

    img_pp = None
    if use_image_features:
        img_pp = ImagePreprocessor(img_use_pca=cfg.get('img_use_pca', True), img_pca_dim=cfg.get('img_pca_dim', 256))
        img_pp.fit([s['X_img'] for s in slides], [s['img_mask'] for s in slides])
        if save_preprocessor_dir is not None:
            img_pp.save(save_preprocessor_dir, img_feature_names=img_feature_names)

    pyg_graphs = []
    for s in slides:
        Xp_gene = pp.transform(s['X'], gene_names)
        if use_image_features:
            Xp_img = img_pp.transform(s['X_img'], s['img_mask'])
            data_g = assemble_pyg(Xp_gene, s['xy'], s['y'], cfg, Xp_img=Xp_img, img_mask=s['img_mask'])
        else:
            data_g = assemble_pyg(Xp_gene, s['xy'], s['y'], cfg)
        data_g.slide_id = str(s['slide_id'])
        # 注入域标签（图级）
        batch_id = _resolve_batch_id(str(s['slide_id']), id2batch, batch_fallback_cache)
        data_g.bat_dom = torch.tensor(batch_to_idx[batch_id], dtype=torch.long)
        ctype = id2type.get(str(s['slide_id']))
        data_g.cancer_dom = torch.tensor(cancer_to_idx[ctype], dtype=torch.long)
        pyg_graphs.append(data_g)

    in_dim = pyg_graphs[0].x.shape[1]

    n_domains_batch = len(batch_to_idx) if cfg.get('use_domain_adv_slide', False) else None
    n_domains_cancer = len(cancer_to_idx) if (cfg.get('use_domain_adv_cancer', False) or cfg.get('use_wb_align', False)) else None

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

    use_image_features = bool(cfg.get('use_image_features', False))
    img_pp = None
    expected_img_feature_names = None
    if use_image_features:
        try:
            img_pp = ImagePreprocessor.load(preprocessor_dir, img_use_pca=cfg.get('img_use_pca', True))
            expected_img_feature_names = img_pp.feature_names
            if not expected_img_feature_names:
                raise ValueError('img_feature_names.txt is missing or empty.')
        except Exception as e:
            print(f"[Warn] Failed to load image preprocessor from {preprocessor_dir}: {e}")
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

        Xp_gene = pp.transform(X, gene_names)
        if use_image_features:
            if {'X_img', 'img_mask'}.issubset(set(d.files)):
                X_img = d['X_img']
                img_mask = d['img_mask']
                if 'img_feature_names' not in d.files:
                    raise ValueError(f"Missing 'img_feature_names' in {npz_path}")
                img_feature_names = list(d['img_feature_names'])
                if expected_img_feature_names is not None and img_feature_names != expected_img_feature_names:
                    raise ValueError(f"img_feature_names mismatch in {npz_path} (sample_id={sample_id})")
            else:
                X_img = np.zeros((X.shape[0], int(img_pp.scaler.mean_.shape[0])), dtype=np.float32)
                img_mask = np.zeros((X.shape[0],), dtype=np.uint8)
            Xp_img = img_pp.transform(X_img, img_mask)
            g = assemble_pyg(Xp_gene, xy, y, cfg, Xp_img=Xp_img, img_mask=img_mask)
        else:
            g = assemble_pyg(Xp_gene, xy, y, cfg)
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
    capture_last_state=False,
    capture_epoch_states=None,
):
    """封装单次训练+验证，返回(best_metrics, hist_dict, best_state_dict)
    report_cb(epoch, metrics) 可选用于HPO报告。
    """
    def _epochs_to_steps(value_epochs, steps_per_epoch):
        value_epochs = int(value_epochs)
        if value_epochs < 0:
            raise ValueError(f'epoch count must be >= 0, got: {value_epochs}')
        return int(value_epochs) * int(steps_per_epoch)

    def _dann_beta(p, beta_target, gamma):
        p = float(p)
        beta_target = float(beta_target)
        gamma = float(gamma)
        return beta_target * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)

    def _dann_beta_with_delay(global_step, total_steps, beta_target, gamma, delay_steps):
        global_step = int(global_step)
        total_steps = int(total_steps)
        delay_steps = int(delay_steps)
        if global_step < delay_steps:
            return 0.0
        rem_steps = max(total_steps - delay_steps, 1)
        p = min(max(float(global_step - delay_steps) / float(rem_steps), 0.0), 1.0)
        return _dann_beta(p, beta_target, gamma)

    def _linear_warmup_beta(global_step, beta_target, delay_steps, warmup_steps):
        global_step = int(global_step)
        delay_steps = int(delay_steps)
        warmup_steps = int(warmup_steps)
        beta_target = float(beta_target)
        if global_step < delay_steps:
            return 0.0
        if warmup_steps <= 0:
            return beta_target
        step_eff = global_step - delay_steps
        progress = min(max(float(step_eff) / float(warmup_steps), 0.0), 1.0)
        return beta_target * progress

    grl_mode = str(cfg.get('grl_beta_mode', 'dann'))
    if grl_mode not in {'dann', 'constant', 'linear'}:
        raise ValueError(f"cfg['grl_beta_mode'] must be 'dann', 'constant' or 'linear', got: {cfg.get('grl_beta_mode')}")
    capture_epoch_states = {int(v) for v in (capture_epoch_states or [])}
    saved_epoch_states = {}

    def _make_graph_frequency_weights(graph_labels, k, device):
        k = int(k)
        if k <= 0:
            return None
        if graph_labels is None:
            return None
        labels = torch.as_tensor(graph_labels, dtype=torch.long)
        if labels.numel() == 0:
            return torch.ones(k, dtype=torch.float32, device=device)
        counts = torch.bincount(labels, minlength=k).to(torch.float32)
        n_graph = float(labels.numel())
        w = torch.sqrt(torch.tensor(n_graph, dtype=torch.float32) / (float(k) * counts.clamp_min(1.0)))
        w = w.clamp(0.5, 5.0)
        w = w / w.mean().clamp_min(1e-6)
        return w.to(device)

    def _sample_nodes_per_graph(h, dom_nodes, graph_nodes, spots_per_slide):
        if spots_per_slide is None or int(spots_per_slide) <= 0:
            return h, dom_nodes
        if graph_nodes is None:
            raise ValueError('graph_nodes is required when mmd_spots_per_slide > 0')

        keep_idx = []
        for gid in sorted(int(v) for v in torch.unique(graph_nodes.detach()).tolist()):
            idx = torch.where(graph_nodes == gid)[0]
            if idx.numel() <= int(spots_per_slide):
                keep_idx.append(idx)
            else:
                perm = torch.randperm(idx.numel(), device=idx.device)[:int(spots_per_slide)]
                keep_idx.append(idx[perm])

        if not keep_idx:
            return h, dom_nodes

        keep_idx = torch.cat(keep_idx, dim=0)
        return h[keep_idx], dom_nodes[keep_idx]

    def _pairwise_sq_dists(x, y):
        return torch.cdist(x, y, p=2).pow(2)

    def _build_sigma_list(x, y, num_kernels=5, kernel_mul=2.0, fixed_sigma=None):
        if fixed_sigma is not None:
            base_sigma = torch.tensor(float(fixed_sigma), dtype=x.dtype, device=x.device)
        else:
            z = torch.cat([x, y], dim=0)
            if z.size(0) <= 1:
                base_sigma = torch.tensor(1.0, dtype=x.dtype, device=x.device)
            else:
                sq = _pairwise_sq_dists(z, z)
                mask = ~torch.eye(sq.size(0), dtype=torch.bool, device=sq.device)
                vals = sq[mask]
                if vals.numel() == 0:
                    base_sigma = torch.tensor(1.0, dtype=x.dtype, device=x.device)
                else:
                    base_sigma = torch.sqrt(vals.mean().clamp_min(1e-12))

        center = int(num_kernels) // 2
        sigma_list = []
        for i in range(int(num_kernels)):
            scale = float(kernel_mul) ** (i - center)
            sigma_list.append((base_sigma * scale).clamp_min(1e-6))
        return sigma_list

    def _rbf_kernel(x, y, sigma_list):
        sq = _pairwise_sq_dists(x, y)
        k = torch.zeros_like(sq)
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma.pow(2).clamp_min(1e-12))
            k = k + torch.exp(-sq * gamma)
        return k

    def _mmd2_unbiased(x, y, sigma_list):
        n = int(x.size(0))
        m = int(y.size(0))
        if n < 2 or m < 2:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        k_xx = _rbf_kernel(x, x, sigma_list)
        k_yy = _rbf_kernel(y, y, sigma_list)
        k_xy = _rbf_kernel(x, y, sigma_list)

        sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / float(n * (n - 1))
        sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / float(m * (m - 1))
        sum_xy = k_xy.mean()
        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
        return torch.clamp(mmd2, min=0.0)

    def _multi_domain_mmd(h, dom_nodes, graph_nodes=None, spots_per_slide=0, num_kernels=5, kernel_mul=2.0, sigma=None, max_pairs=8):
        if h is None or dom_nodes is None:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        if h.size(0) != dom_nodes.size(0):
            raise ValueError(f"MMD shape mismatch: h={tuple(h.shape)}, dom_nodes={tuple(dom_nodes.shape)}")

        h, dom_nodes = _sample_nodes_per_graph(h, dom_nodes, graph_nodes, spots_per_slide)

        groups = {}
        for dom in sorted(int(v) for v in torch.unique(dom_nodes.detach()).tolist()):
            mask = dom_nodes == dom
            if int(mask.sum().item()) >= 2:
                groups[dom] = h[mask]

        pairs = list(itertools.combinations(sorted(groups.keys()), 2))
        if max_pairs is not None and int(max_pairs) > 0:
            pairs = pairs[:int(max_pairs)]

        if not pairs:
            return torch.tensor(0.0, dtype=h.dtype, device=h.device)

        losses = []
        for da, db in pairs:
            xa = groups[da]
            xb = groups[db]
            sigma_list = _build_sigma_list(
                xa,
                xb,
                num_kernels=num_kernels,
                kernel_mul=kernel_mul,
                fixed_sigma=sigma,
            )
            losses.append(_mmd2_unbiased(xa, xb, sigma_list))

        return torch.stack(losses).mean()

    def _normalize_int_cfg(name, value, default_value):
        if value is None:
            return int(default_value)
        if isinstance(value, bool):
            raise ValueError(f"cfg['{name}'] must be an integer, got bool: {value}")
        if isinstance(value, float) and not float(value).is_integer():
            raise ValueError(f"cfg['{name}'] must be an integer, got: {value}")
        return int(value)

    def _plateau_mode_from_metric(metric_name):
        if metric_name == 'val_avg_total_loss':
            return 'min'
        return 'max'

    def _build_step_lr_scheduler(optimizer, scheduler_mode, total_steps, warmup_steps, min_lr_ratio):
        if int(total_steps) <= 0:
            return None

        warmup_steps = max(0, int(warmup_steps))
        eps = 1e-8

        def _lr_scale(step_idx):
            completed_steps = max(0, int(step_idx) + 1)
            if scheduler_mode == 'linear':
                progress = min(max(float(completed_steps) / float(max(int(total_steps), 1)), 0.0), 1.0)
                return 1.0 - (1.0 - float(min_lr_ratio)) * progress
            if scheduler_mode == 'cosine':
                progress = min(max(float(completed_steps) / float(max(int(total_steps), 1)), 0.0), 1.0)
                return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if scheduler_mode == 'warmup_cosine':
                if warmup_steps <= 0:
                    progress = min(max(float(completed_steps) / float(max(int(total_steps), 1)), 0.0), 1.0)
                    return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * 0.5 * (1.0 + math.cos(math.pi * progress))
                if warmup_steps >= int(total_steps):
                    progress = min(max(float(completed_steps) / float(max(int(total_steps), 1)), 0.0), 1.0)
                    return max(eps, progress)
                if completed_steps <= warmup_steps:
                    progress = min(max(float(completed_steps) / float(max(warmup_steps, 1)), 0.0), 1.0)
                    return max(eps, progress)
                decay_steps = max(int(total_steps) - warmup_steps, 1)
                decay_progress = min(max(float(completed_steps - warmup_steps) / float(decay_steps), 0.0), 1.0)
                return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_scale)
        if scheduler_mode == 'warmup_cosine' and warmup_steps > 0:
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group['lr'] = float(base_lr) * eps
        return scheduler

    def _set_requires_grad(module_or_params, enabled):
        if module_or_params is None:
            return
        params = module_or_params.parameters() if hasattr(module_or_params, 'parameters') else module_or_params
        for p in params:
            p.requires_grad_(bool(enabled))

    def _wb_lambda_for_step(step_idx):
        if not bool(cfg.get('use_wb_align', False)):
            return 0.0
        lambda_max = float(cfg.get('lambda_wb', 0.01))
        warmup_steps = _epochs_to_steps(cfg.get('wb_warmup_epochs', 10), steps_per_epoch)
        ramp_steps = _epochs_to_steps(cfg.get('wb_ramp_epochs', 20), steps_per_epoch)
        if int(step_idx) < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return lambda_max
        progress = min(max(float(int(step_idx) - warmup_steps) / float(max(ramp_steps, 1)), 0.0), 1.0)
        return lambda_max * progress

    def _stat_to_float(stats, key, default=float('nan')):
        value = stats.get(key, default) if isinstance(stats, dict) else default
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            return float(value.detach().mean().item())
        try:
            return float(value)
        except Exception:
            return default

    model = STOnco_Classifier(
        in_dim=in_dim,
        hidden=cfg['GNN_hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
        clf_hidden=cfg.get('clf_hidden', [256, 128, 64]),
        domain_hidden=int(cfg.get('dom_hidden', 64)),
        use_domain_adv_slide=cfg.get('use_domain_adv_slide', False), n_domains_slide=n_domains_batch,
        use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
    )
    model = model.to(device)
    use_wb_align = bool(cfg.get('use_wb_align', False))
    support_map = None
    wb_module = None
    opt_pot = None
    if use_wb_align:
        if n_domains_cancer is None or int(n_domains_cancer) <= 0:
            raise ValueError('use_wb_align=1 requires n_domains_cancer; prepare_graphs must return it when WB is enabled.')
        support_map = GeneratedSupportMap(
            model.gnn.out_dim,
            hidden=int(cfg.get('wb_support_hidden', 128)),
            dropout=float(cfg.get('wb_support_dropout', 0.0)),
        ).to(device)
        wb_module = GeneratedSupportWBLoss(
            n_domains=int(n_domains_cancer),
            in_dim=model.gnn.out_dim,
            loss_type=str(cfg.get('wb_loss_type', 'euclidean_pairwise')),
            potential_hidden=int(cfg.get('wb_potential_hidden', 128)),
            spots_per_graph=int(cfg.get('wb_spots_per_graph', 64)),
            spots_per_cancer=int(cfg.get('wb_spots_per_cancer', 0)),
            support_size=int(cfg.get('wb_support_size', 128)),
            min_cancers=int(cfg.get('wb_min_cancers', 2)),
            min_spots=int(cfg.get('wb_min_spots', 2)),
            regularizer=str(cfg.get('wb_regularizer', 'l2')),
            epsilon=float(cfg.get('wb_epsilon', 0.1)),
            label_balanced_sampling=bool(cfg.get('wb_label_balanced_sampling', False)),
            state_direction=bool(cfg.get('wb_state_direction', False)),
            state_direction_weight=float(cfg.get('wb_state_direction_weight', 0.1)),
            euclid_pairwise_weight=float(cfg.get('wb_euclid_pairwise_weight', 1.0)),
            potential_weight=float(cfg.get('wb_potential_weight', 1.0)),
            potential_constraint_weight=float(cfg.get('wb_potential_constraint_weight', 0.01)),
        ).to(device)
        opt_pot = torch.optim.AdamW(
            list(wb_module.potential_parameters()),
            lr=float(cfg.get('wb_potential_lr', 1e-4)),
            weight_decay=0.0,
        )
    main_params = list(model.parameters())
    if support_map is not None:
        main_params += list(support_map.parameters())
    if wb_module is not None:
        main_params += list(wb_module.main_parameters())
    opt = torch.optim.AdamW(main_params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    bce = nn.BCEWithLogitsLoss()

    lr_scheduler_name = str(cfg.get('lr_scheduler', 'none')).lower()
    if lr_scheduler_name not in {'none', 'linear', 'cosine', 'warmup_cosine', 'plateau'}:
        raise ValueError(f"cfg['lr_scheduler'] must be one of none/linear/cosine/warmup_cosine/plateau, got: {cfg.get('lr_scheduler')}")
    base_lr = float(cfg.get('lr', 0.0))
    if base_lr <= 0:
        raise ValueError(f"cfg['lr'] must be > 0, got: {cfg.get('lr')}")
    min_lr_ratio = float(cfg.get('min_lr_ratio', 0.01))
    if not (0.0 < min_lr_ratio <= 1.0):
        raise ValueError(f"cfg['min_lr_ratio'] must be in (0, 1], got: {min_lr_ratio}")
    lr_warmup_epochs = _normalize_int_cfg('lr_warmup_epochs', cfg.get('lr_warmup_epochs', 10), 10)
    plateau_metric = str(cfg.get('plateau_metric', 'val_accuracy'))
    if plateau_metric not in {'val_accuracy', 'val_avg_total_loss', 'val_macro_f1', 'val_auroc', 'val_auprc'}:
        raise ValueError(f"cfg['plateau_metric'] must be one of val_accuracy/val_avg_total_loss/val_macro_f1/val_auroc/val_auprc, got: {plateau_metric}")
    plateau_factor = float(cfg.get('plateau_factor', 0.5))
    if not (0.0 < plateau_factor < 1.0):
        raise ValueError(f"cfg['plateau_factor'] must be in (0, 1), got: {plateau_factor}")
    plateau_patience = _normalize_int_cfg('plateau_patience', cfg.get('plateau_patience', 10), 10)
    plateau_cooldown = _normalize_int_cfg('plateau_cooldown', cfg.get('plateau_cooldown', 0), 0)
    plateau_threshold = float(cfg.get('plateau_threshold', 1e-4))
    if plateau_patience < 0:
        raise ValueError(f"cfg['plateau_patience'] must be >= 0, got: {plateau_patience}")
    if plateau_cooldown < 0:
        raise ValueError(f"cfg['plateau_cooldown'] must be >= 0, got: {plateau_cooldown}")
    if plateau_threshold < 0:
        raise ValueError(f"cfg['plateau_threshold'] must be >= 0, got: {plateau_threshold}")

    effective_train_graphs = build_training_subgraphs(train_graphs, cfg)
    if str(cfg.get('subgraph_mode', 'off')).lower() != 'off':
        print(
            f"[Sampler] subgraph_mode={cfg.get('subgraph_mode')}: expanded "
            f"{len(train_graphs)} parent graphs to {len(effective_train_graphs)} training graphs"
        )

    batch_size_graphs = int(cfg['batch_size_graphs'])
    sampler_mode = str(cfg.get('sampler_mode', 'random')).lower()
    use_balanced_sampler = (
        sampler_mode == 'cancer_balanced'
        and batch_size_graphs >= 4
        and len(effective_train_graphs) >= batch_size_graphs
    )

    if use_balanced_sampler:
        train_batch_sampler = CancerBalancedBatchSampler(
            effective_train_graphs,
            batch_size=batch_size_graphs,
            k_cancers=int(cfg.get('sampler_k_cancers', 1)),
            m_per_cancer=int(cfg.get('sampler_m_per_cancer', 1)),
            seed=int(cfg.get('sampler_seed', 42)),
            enforce_distinct_batch=bool(cfg.get('sampler_enforce_distinct_batch', True)),
            num_batches=int(math.ceil(len(effective_train_graphs) / float(max(batch_size_graphs, 1)))),
        )
        sampler_preview = train_batch_sampler.preview_batches()
        sampler_summary = summarize_batches(effective_train_graphs, sampler_preview)
        print(
            f"[Sampler] mode={sampler_mode}, K={cfg.get('sampler_k_cancers')}, M={cfg.get('sampler_m_per_cancer')}, "
            f"batch_size={batch_size_graphs}, avg_unique_cancers={sampler_summary['avg_unique_cancers']:.2f}, "
            f"avg_unique_batches={sampler_summary['avg_unique_batches']:.2f}, "
            f"avg_unique_slides={sampler_summary['avg_unique_slides']:.2f}, "
            f"avg_unique_parents={sampler_summary['avg_unique_parents']:.2f}"
        )
        train_loader = PyGDataLoader(effective_train_graphs, batch_sampler=train_batch_sampler, num_workers=num_workers)
    else:
        if sampler_mode == 'cancer_balanced' and batch_size_graphs < 4:
            print(
                f"[Sampler] Warning: batch_size_graphs={batch_size_graphs} is too small for cancer-balanced batching; "
                "fallback to random shuffle."
            )
        train_loader = PyGDataLoader(effective_train_graphs, batch_size=batch_size_graphs, shuffle=True, num_workers=num_workers)

    val_loader = PyGDataLoader(
        val_graphs,
        batch_size=cfg['batch_size_graphs'],
        shuffle=False,
        num_workers=max(0, min(num_workers, 2)),
    )
    extra_val_graphs = list(external_val_graphs) if external_val_graphs else []

    # domain class weights (graph-frequency, sqrt inverse freq; clamp + mean-normalize)
    dom_w_batch = None
    dom_w_cancer = None
    if cfg.get('use_domain_adv_slide', False) and n_domains_batch is not None:
        dom_w_batch = _make_graph_frequency_weights(
            [int(g.bat_dom.item()) for g in effective_train_graphs if hasattr(g, 'bat_dom')],
            int(n_domains_batch),
            device,
        )
    if cfg.get('use_domain_adv_cancer', False) and n_domains_cancer is not None:
        dom_w_cancer = _make_graph_frequency_weights(
            [int(g.cancer_dom.item()) for g in effective_train_graphs if hasattr(g, 'cancer_dom')],
            int(n_domains_cancer),
            device,
        )
    cel_batch = nn.CrossEntropyLoss(weight=dom_w_batch) if dom_w_batch is not None else nn.CrossEntropyLoss()
    cel_cancer = nn.CrossEntropyLoss(weight=dom_w_cancer) if dom_w_cancer is not None else nn.CrossEntropyLoss()

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
        loss_mmd = torch.tensor(0.0, device=device)
        ce_batch = torch.tensor(float('nan'), device=device)
        ce_cancer = torch.tensor(float('nan'), device=device)
        raw_mmd_slide = torch.tensor(float('nan'), device=device)
        raw_mmd_cancer = torch.tensor(float('nan'), device=device)

        if out.get('dom_logits_slide', None) is not None and hasattr(batch, 'bat_dom') and hasattr(batch, 'batch'):
            if n_domains_batch is None:
                raise ValueError('n_domains_batch is None while batch domain logits are enabled.')
            dom_nodes = batch.bat_dom[batch.batch]
            dom_min = int(dom_nodes.min().item())
            dom_max = int(dom_nodes.max().item())
            if dom_min < 0 or dom_max >= int(n_domains_batch):
                raise ValueError(
                    f"[DomainCheck] batch_dom out of range: min={dom_min}, max={dom_max}, "
                    f"n_domains_batch={n_domains_batch}, slide_ids={getattr(batch, 'slide_id', 'NA')}"
                )
            ce_batch = F.cross_entropy(out['dom_logits_slide'], dom_nodes)
            loss_dom_batch = cel_batch(out['dom_logits_slide'], dom_nodes)
            loss_batch = float(cfg.get('lambda_slide', 1.0)) * loss_dom_batch
            total = total + loss_batch
        if out.get('dom_logits_cancer', None) is not None and hasattr(batch, 'cancer_dom') and hasattr(batch, 'batch'):
            if n_domains_cancer is None:
                raise ValueError('n_domains_cancer is None while cancer domain logits are enabled.')
            dom_nodes = batch.cancer_dom[batch.batch]
            dom_min = int(dom_nodes.min().item())
            dom_max = int(dom_nodes.max().item())
            if dom_min < 0 or dom_max >= int(n_domains_cancer):
                raise ValueError(
                    f"[DomainCheck] cancer_dom out of range: min={dom_min}, max={dom_max}, "
                    f"n_domains_cancer={n_domains_cancer}, slide_ids={getattr(batch, 'slide_id', 'NA')}"
                )
            ce_cancer = F.cross_entropy(out['dom_logits_cancer'], dom_nodes)
            loss_dom_cancer = cel_cancer(out['dom_logits_cancer'], dom_nodes)
            loss_cancer = float(cfg.get('lambda_cancer', 1.0)) * loss_dom_cancer
            total = total + loss_cancer

        if bool(cfg.get('use_mmd', False)):
            h = out.get('h', None)
            if h is None:
                raise ValueError("MMD is enabled but model output does not contain 'h'.")

            mmd_on = str(cfg.get('mmd_on', 'slide')).lower()
            lambda_mmd = float(cfg.get('lambda_mmd', 0.05))
            num_kernels = int(cfg.get('mmd_num_kernels', 5))
            kernel_mul = float(cfg.get('mmd_kernel_mul', 2.0))
            sigma = cfg.get('mmd_sigma', None)
            max_pairs = int(cfg.get('mmd_max_pairs', 8))
            spots_per_slide = int(cfg.get('mmd_spots_per_slide', 0))

            if mmd_on in {'slide', 'both'} and hasattr(batch, 'bat_dom') and hasattr(batch, 'batch'):
                dom_nodes_slide = batch.bat_dom[batch.batch]
                raw_mmd_slide = _multi_domain_mmd(
                    h,
                    dom_nodes_slide,
                    graph_nodes=getattr(batch, 'batch', None),
                    spots_per_slide=spots_per_slide,
                    num_kernels=num_kernels,
                    kernel_mul=kernel_mul,
                    sigma=sigma,
                    max_pairs=max_pairs,
                )
                loss_mmd = loss_mmd + lambda_mmd * raw_mmd_slide

            if mmd_on in {'cancer', 'both'} and hasattr(batch, 'cancer_dom') and hasattr(batch, 'batch'):
                dom_nodes_cancer = batch.cancer_dom[batch.batch]
                raw_mmd_cancer = _multi_domain_mmd(
                    h,
                    dom_nodes_cancer,
                    graph_nodes=getattr(batch, 'batch', None),
                    spots_per_slide=spots_per_slide,
                    num_kernels=num_kernels,
                    kernel_mul=kernel_mul,
                    sigma=sigma,
                    max_pairs=max_pairs,
                )
                loss_mmd = loss_mmd + lambda_mmd * raw_mmd_cancer

            total = total + loss_mmd

        return (
            total,
            loss_task,
            loss_batch,
            loss_cancer,
            loss_mmd,
            ce_batch,
            ce_cancer,
            raw_mmd_slide,
            raw_mmd_cancer,
        )

    best = {'auroc': -1, 'accuracy': -1, 'state': None, 'epoch': -1, 'macro_f1': float('nan'), 'auprc': float('nan')}
    last_epoch = 0
    last_metrics = None
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
        'avg_batch_domain_ce': [],
        'avg_cancer_domain_ce': [],
        'avg_mmd_loss': [],
        'avg_mmd_raw_slide': [],
        'avg_mmd_raw_cancer': [],
        'avg_wb_loss': [],
        'avg_wb_potential_loss': [],
        'avg_wb_dual_obj': [],
        'avg_wb_euclid_pairwise': [],
        'avg_wb_anchor': [],
        'avg_wb_state_direction': [],
        'avg_wb_active_cancers': [],
        'avg_wb_active_spots': [],
        'wb_lambda': [],
        'train_batch_domain_acc': [],
        'train_cancer_domain_acc': [],
        # 2026-04 update: rename the former "var_risk" metric to
        # "batch_loss_variance" to match its actual definition.
        'batch_loss_variance': [],
        'train_accuracy': [],
        'val_avg_total_loss': [],
        'val_avg_task_loss': [],
        'val_avg_batch_domain_loss': [],
        'val_avg_cancer_domain_loss': [],
        'val_avg_batch_domain_ce': [],
        'val_avg_cancer_domain_ce': [],
        'val_avg_mmd_loss': [],
        'val_avg_mmd_raw_slide': [],
        'val_avg_mmd_raw_cancer': [],
        'val_avg_wb_loss': [],
        'val_avg_wb_anchor': [],
        'val_avg_wb_state_direction': [],
        'val_avg_wb_active_cancers': [],
        'val_avg_wb_active_spots': [],
        'val_batch_loss_variance': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'val_auroc': [],
        'val_auprc': [],
        'lr': [],
    }

    epoch_iter = range(1, cfg['epochs'] + 1)
    if progress_desc:
        epoch_iter = tqdm(epoch_iter, desc=progress_desc, leave=progress_leave)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = int(cfg['epochs']) * steps_per_epoch
    lr_warmup_steps = _epochs_to_steps(max(0, lr_warmup_epochs), steps_per_epoch)
    lr_scheduler = None
    if lr_scheduler_name in {'linear', 'cosine', 'warmup_cosine'}:
        lr_scheduler = _build_step_lr_scheduler(
            opt,
            scheduler_mode=lr_scheduler_name,
            total_steps=total_steps,
            warmup_steps=lr_warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        if lr_scheduler is None:
            print(f"[Warn] lr_scheduler='{lr_scheduler_name}' skipped because total_steps={total_steps}. Falling back to fixed lr.")
    elif lr_scheduler_name == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=_plateau_mode_from_metric(plateau_metric),
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            cooldown=plateau_cooldown,
            min_lr=base_lr * min_lr_ratio,
        )
        if early_stop_enabled:
            recommended_patience = max(plateau_patience + plateau_cooldown + 3, 2 * max(plateau_patience, 1))
            print(
                f"[Warn] lr_scheduler='plateau' is enabled together with early stopping "
                f"(early_patience={early_patience}, plateau_patience={plateau_patience}, plateau_cooldown={plateau_cooldown}). "
                f"These controls may compete. Recommend disabling early stopping (--early_patience 0), "
                f"or increasing early_patience to at least {recommended_patience}."
            )
    slide_delay_steps = _epochs_to_steps(cfg.get('grl_beta_slide_delay_epochs', 1), steps_per_epoch)
    slide_warmup_steps = _epochs_to_steps(cfg.get('grl_beta_slide_warmup_epochs', 8), steps_per_epoch)
    cancer_delay_steps = _epochs_to_steps(cfg.get('grl_beta_cancer_delay_epochs', 3), steps_per_epoch)
    cancer_warmup_steps = _epochs_to_steps(cfg.get('grl_beta_cancer_warmup_epochs', 12), steps_per_epoch)
    global_step = 0
    for epoch in epoch_iter:
        model.train()
        if support_map is not None:
            support_map.train()
        if wb_module is not None:
            wb_module.train()
        tot_total = 0.0
        tot_task = 0.0
        tot_batch = 0.0
        tot_cancer = 0.0
        tot_batch_ce = 0.0
        tot_cancer_ce = 0.0
        tot_mmd = 0.0
        tot_mmd_raw_slide = 0.0
        tot_mmd_raw_cancer = 0.0
        cnt_mmd_slide = 0
        cnt_mmd_cancer = 0
        tot_wb = 0.0
        tot_wb_potential = 0.0
        tot_wb_dual = 0.0
        tot_wb_euclid = 0.0
        tot_wb_anchor = 0.0
        tot_wb_state = 0.0
        tot_wb_active_cancers = 0.0
        tot_wb_active_spots = 0.0
        tot_wb_lambda = 0.0
        cnt_wb = 0
        cnt_wb_potential = 0
        cnt_wb_dual = 0
        cnt_wb_euclid = 0
        batch_losses = []
        num_batches = 0
        train_correct = 0
        train_total = 0
        dom_slide_correct = 0
        dom_slide_total = 0
        dom_cancer_correct = 0
        dom_cancer_total = 0
        for batch in train_loader:
            batch = batch.to(device)
            wb_lambda_t = _wb_lambda_for_step(global_step)
            if use_wb_align and wb_module is not None and support_map is not None and wb_lambda_t > 0.0:
                if not hasattr(batch, 'cancer_dom') or not hasattr(batch, 'batch'):
                    raise ValueError('use_wb_align=1 requires batch.cancer_dom and batch.batch for per-spot cancer labels.')
                if int(global_step) % int(cfg.get('wb_pot_every_n_steps', 1)) == 0:
                    _set_requires_grad(model, False)
                    _set_requires_grad(support_map, False)
                    _set_requires_grad(wb_module.main_parameters(), False)
                    _set_requires_grad(wb_module.potential_parameters(), True)
                    opt_pot.zero_grad()
                    with torch.no_grad():
                        h_detached = model.gnn(
                            batch.x,
                            batch.edge_index,
                            edge_weight=getattr(batch, 'edge_weight', None),
                        )
                        b_detached = support_map(h_detached).detach()
                    dom_nodes_wb = batch.cancer_dom[batch.batch]
                    pot_loss, pot_stats = wb_module.potential_loss(
                        h=h_detached.detach(),
                        b=b_detached,
                        cancer_dom=dom_nodes_wb,
                        graph_nodes=getattr(batch, 'batch', None),
                        y=getattr(batch, 'y', None),
                    )
                    if bool(pot_stats.get('valid', False)) and pot_loss.requires_grad:
                        pot_loss.backward()
                        opt_pot.step()
                        tot_wb_potential += float(pot_loss.detach().item())
                        cnt_wb_potential += 1

            _set_requires_grad(model, True)
            _set_requires_grad(support_map, True)
            if wb_module is not None:
                _set_requires_grad(wb_module.main_parameters(), True)
                _set_requires_grad(wb_module.potential_parameters(), False)
            opt.zero_grad()
            if grl_mode == 'constant':
                grl_beta_slide = float(cfg.get('grl_beta_slide_target', 1.0))
                grl_beta_cancer = float(cfg.get('grl_beta_cancer_target', 0.5))
            elif grl_mode == 'dann':
                beta_gamma = float(cfg.get('grl_beta_gamma', 10.0))
                grl_beta_slide = _dann_beta_with_delay(
                    global_step,
                    total_steps,
                    float(cfg.get('grl_beta_slide_target', 1.0)),
                    beta_gamma,
                    slide_delay_steps,
                )
                grl_beta_cancer = _dann_beta_with_delay(
                    global_step,
                    total_steps,
                    float(cfg.get('grl_beta_cancer_target', 0.5)),
                    beta_gamma,
                    cancer_delay_steps,
                )
            else:
                grl_beta_slide = _linear_warmup_beta(
                    global_step,
                    float(cfg.get('grl_beta_slide_target', 1.0)),
                    slide_delay_steps,
                    slide_warmup_steps,
                )
                grl_beta_cancer = _linear_warmup_beta(
                    global_step,
                    float(cfg.get('grl_beta_cancer_target', 0.5)),
                    cancer_delay_steps,
                    cancer_warmup_steps,
                )
            out = model(
                batch.x,
                batch.edge_index,
                batch=getattr(batch, 'batch', None),
                edge_weight=getattr(batch, 'edge_weight', None),
                grl_beta_slide=grl_beta_slide,
                grl_beta_cancer=grl_beta_cancer,
                return_h=bool(cfg.get('use_mmd', False) or cfg.get('use_wb_align', False)),
            )
            global_step += 1

            # 域头训练准确率（spot-level；不参与反传）
            with torch.no_grad():
                if out.get('dom_logits_slide', None) is not None and hasattr(batch, 'bat_dom') and hasattr(batch, 'batch'):
                    pred = out['dom_logits_slide'].argmax(dim=1)
                    tgt = batch.bat_dom[batch.batch]
                    dom_slide_correct += int((pred == tgt).sum().item())
                    dom_slide_total += int(tgt.numel())
                if out.get('dom_logits_cancer', None) is not None and hasattr(batch, 'cancer_dom') and hasattr(batch, 'batch'):
                    pred = out['dom_logits_cancer'].argmax(dim=1)
                    tgt = batch.cancer_dom[batch.batch]
                    dom_cancer_correct += int((pred == tgt).sum().item())
                    dom_cancer_total += int(tgt.numel())

            (
                loss_total,
                loss_task,
                loss_batch,
                loss_cancer,
                loss_mmd,
                ce_batch,
                ce_cancer,
                raw_mmd_slide,
                raw_mmd_cancer,
            ) = _compute_losses(out, batch)
            loss_wb = torch.tensor(0.0, device=device)
            loss_wb_anchor = torch.tensor(0.0, device=device)
            wb_stats = {'valid': False}
            if use_wb_align and wb_module is not None and support_map is not None and wb_lambda_t > 0.0:
                h = out.get('h', None)
                if h is None:
                    raise ValueError("WB is enabled but model output does not contain 'h'.")
                if not hasattr(batch, 'cancer_dom') or not hasattr(batch, 'batch'):
                    raise ValueError('use_wb_align=1 requires batch.cancer_dom and batch.batch for per-spot cancer labels.')
                b_wb = support_map(h)
                dom_nodes_wb = batch.cancer_dom[batch.batch]
                loss_wb, loss_wb_anchor, wb_stats = wb_module.model_loss(
                    h=h,
                    b=b_wb,
                    cancer_dom=dom_nodes_wb,
                    graph_nodes=getattr(batch, 'batch', None),
                    y=getattr(batch, 'y', None),
                )
                loss_total = loss_total + float(wb_lambda_t) * (
                    loss_wb + float(cfg.get('wb_anchor_weight', 0.5)) * loss_wb_anchor
                )
            loss_total.backward()
            opt.step()
            if wb_module is not None:
                _set_requires_grad(wb_module.potential_parameters(), True)
            if lr_scheduler is not None and lr_scheduler_name in {'linear', 'cosine', 'warmup_cosine'}:
                lr_scheduler.step()

            tot_total += float(loss_total.item())
            tot_task += float(loss_task.item())
            tot_batch += float(loss_batch.item())
            tot_cancer += float(loss_cancer.item())
            tot_mmd += float(loss_mmd.item())
            if torch.isfinite(ce_batch):
                tot_batch_ce += float(ce_batch.item())
            if torch.isfinite(ce_cancer):
                tot_cancer_ce += float(ce_cancer.item())
            if torch.isfinite(raw_mmd_slide):
                tot_mmd_raw_slide += float(raw_mmd_slide.item())
                cnt_mmd_slide += 1
            if torch.isfinite(raw_mmd_cancer):
                tot_mmd_raw_cancer += float(raw_mmd_cancer.item())
                cnt_mmd_cancer += 1
            if use_wb_align:
                tot_wb_lambda += float(wb_lambda_t)
                if bool(wb_stats.get('valid', False)):
                    tot_wb += float(loss_wb.detach().item())
                    tot_wb_anchor += float(loss_wb_anchor.detach().item())
                    tot_wb_active_cancers += _stat_to_float(wb_stats, 'wb_active_cancers', default=0.0)
                    tot_wb_active_spots += _stat_to_float(wb_stats, 'wb_active_spots', default=0.0)
                    wb_state = _stat_to_float(wb_stats, 'wb_state_direction', default=float('nan'))
                    if np.isfinite(wb_state):
                        tot_wb_state += wb_state
                    wb_dual = _stat_to_float(wb_stats, 'wb_dual_obj', default=float('nan'))
                    if np.isfinite(wb_dual):
                        tot_wb_dual += wb_dual
                        cnt_wb_dual += 1
                    wb_euclid = _stat_to_float(wb_stats, 'wb_euclid_pairwise', default=float('nan'))
                    if np.isfinite(wb_euclid):
                        tot_wb_euclid += wb_euclid
                        cnt_wb_euclid += 1
                    cnt_wb += 1
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
        avg_mmd = tot_mmd / max(1, num_batches)
        avg_batch_ce = (tot_batch_ce / max(1, num_batches)) if cfg.get('use_domain_adv_slide', False) else float('nan')
        avg_cancer_ce = (tot_cancer_ce / max(1, num_batches)) if cfg.get('use_domain_adv_cancer', False) else float('nan')
        avg_mmd_raw_slide = (tot_mmd_raw_slide / cnt_mmd_slide) if cnt_mmd_slide > 0 else float('nan')
        avg_mmd_raw_cancer = (tot_mmd_raw_cancer / cnt_mmd_cancer) if cnt_mmd_cancer > 0 else float('nan')
        avg_wb = (tot_wb / cnt_wb) if cnt_wb > 0 else float('nan')
        avg_wb_potential = (tot_wb_potential / cnt_wb_potential) if cnt_wb_potential > 0 else float('nan')
        avg_wb_dual = (tot_wb_dual / cnt_wb_dual) if cnt_wb_dual > 0 else float('nan')
        avg_wb_euclid = (tot_wb_euclid / cnt_wb_euclid) if cnt_wb_euclid > 0 else float('nan')
        avg_wb_anchor = (tot_wb_anchor / cnt_wb) if cnt_wb > 0 else float('nan')
        avg_wb_state = (tot_wb_state / cnt_wb) if cnt_wb > 0 else float('nan')
        avg_wb_active_cancers = (tot_wb_active_cancers / cnt_wb) if cnt_wb > 0 else float('nan')
        avg_wb_active_spots = (tot_wb_active_spots / cnt_wb) if cnt_wb > 0 else float('nan')
        avg_wb_lambda = (tot_wb_lambda / max(1, num_batches)) if use_wb_align else float('nan')
        if use_wb_align and avg_wb_lambda > 0 and cnt_wb == 0:
            print(
                f"[WB] Warning: epoch {epoch} produced no valid WB batches. "
                f"Need at least wb_min_cancers={cfg.get('wb_min_cancers')} and "
                f"wb_min_spots={cfg.get('wb_min_spots')} per active cancer after spot sampling."
            )
        train_batch_domain_acc = float(dom_slide_correct / dom_slide_total) if dom_slide_total > 0 else float('nan')
        train_cancer_domain_acc = float(dom_cancer_correct / dom_cancer_total) if dom_cancer_total > 0 else float('nan')
        # batch_loss_variance is the per-epoch variance of mini-batch total loss.
        batch_loss_variance = float(np.mean([(v - avg_total) ** 2 for v in batch_losses])) if batch_losses else float('nan')
        train_accuracy = float(train_correct / train_total) if train_total > 0 else float('nan')

        hist['avg_total_loss'].append(avg_total)
        hist['avg_task_loss'].append(avg_task)
        hist['avg_batch_domain_loss'].append(avg_batch)
        hist['avg_cancer_domain_loss'].append(avg_cancer)
        hist['avg_batch_domain_ce'].append(avg_batch_ce)
        hist['avg_cancer_domain_ce'].append(avg_cancer_ce)
        hist['avg_mmd_loss'].append(avg_mmd)
        hist['avg_mmd_raw_slide'].append(avg_mmd_raw_slide)
        hist['avg_mmd_raw_cancer'].append(avg_mmd_raw_cancer)
        hist['avg_wb_loss'].append(avg_wb)
        hist['avg_wb_potential_loss'].append(avg_wb_potential)
        hist['avg_wb_dual_obj'].append(avg_wb_dual)
        hist['avg_wb_euclid_pairwise'].append(avg_wb_euclid)
        hist['avg_wb_anchor'].append(avg_wb_anchor)
        hist['avg_wb_state_direction'].append(avg_wb_state)
        hist['avg_wb_active_cancers'].append(avg_wb_active_cancers)
        hist['avg_wb_active_spots'].append(avg_wb_active_spots)
        hist['wb_lambda'].append(avg_wb_lambda)
        hist['train_batch_domain_acc'].append(train_batch_domain_acc)
        hist['train_cancer_domain_acc'].append(train_cancer_domain_acc)
        hist['batch_loss_variance'].append(batch_loss_variance)
        hist['train_accuracy'].append(train_accuracy)

        # 验证（内部 + 外部）
        model.eval()
        if support_map is not None:
            support_map.eval()
        if wb_module is not None:
            wb_module.eval()
        val_logits_list = []
        val_y_list = []
        per_slide_acc = []
        val_tot_total = 0.0
        val_tot_task = 0.0
        val_tot_batch = 0.0
        val_tot_cancer = 0.0
        val_tot_batch_ce = 0.0
        val_tot_cancer_ce = 0.0
        val_tot_mmd = 0.0
        val_tot_mmd_raw_slide = 0.0
        val_tot_mmd_raw_cancer = 0.0
        val_cnt_mmd_slide = 0
        val_cnt_mmd_cancer = 0
        val_tot_wb = 0.0
        val_tot_wb_anchor = 0.0
        val_tot_wb_state = 0.0
        val_tot_wb_active_cancers = 0.0
        val_tot_wb_active_spots = 0.0
        val_cnt_wb = 0
        val_num_batches = 0
        val_batch_losses = []
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                out_v = model(
                    vb.x,
                    vb.edge_index,
                    batch=getattr(vb, 'batch', None),
                    edge_weight=getattr(vb, 'edge_weight', None),
                    return_h=bool(cfg.get('use_mmd', False) or (cfg.get('use_wb_align', False) and cfg.get('wb_eval_loss', False))),
                )
                (
                    val_loss_total,
                    val_loss_task,
                    val_loss_batch,
                    val_loss_cancer,
                    val_loss_mmd,
                    val_ce_batch,
                    val_ce_cancer,
                    val_raw_mmd_slide,
                    val_raw_mmd_cancer,
                ) = _compute_losses(out_v, vb)
                if use_wb_align and bool(cfg.get('wb_eval_loss', False)) and wb_module is not None and support_map is not None:
                    h_v = out_v.get('h', None)
                    if h_v is not None and hasattr(vb, 'cancer_dom') and hasattr(vb, 'batch'):
                        b_v = support_map(h_v)
                        val_loss_wb, val_loss_wb_anchor, val_wb_stats = wb_module.model_loss(
                            h=h_v,
                            b=b_v,
                            cancer_dom=vb.cancer_dom[vb.batch],
                            graph_nodes=getattr(vb, 'batch', None),
                            y=getattr(vb, 'y', None),
                        )
                        if bool(val_wb_stats.get('valid', False)):
                            val_loss_total = val_loss_total + float(_wb_lambda_for_step(global_step)) * (
                                val_loss_wb + float(cfg.get('wb_anchor_weight', 0.5)) * val_loss_wb_anchor
                            )
                            val_tot_wb += float(val_loss_wb.detach().item())
                            val_tot_wb_anchor += float(val_loss_wb_anchor.detach().item())
                            val_state = _stat_to_float(val_wb_stats, 'wb_state_direction', default=float('nan'))
                            if np.isfinite(val_state):
                                val_tot_wb_state += val_state
                            val_tot_wb_active_cancers += _stat_to_float(val_wb_stats, 'wb_active_cancers', default=0.0)
                            val_tot_wb_active_spots += _stat_to_float(val_wb_stats, 'wb_active_spots', default=0.0)
                            val_cnt_wb += 1
                logits_v = out_v['logits']
                val_logits_list.append(logits_v.cpu())
                val_y_list.append(vb.y.cpu())
                m_slide = eval_logits(logits_v, vb.y)
                per_slide_acc.append(m_slide.get('accuracy', float('nan')))
                val_tot_total += float(val_loss_total.item())
                val_tot_task += float(val_loss_task.item())
                val_tot_batch += float(val_loss_batch.item())
                val_tot_cancer += float(val_loss_cancer.item())
                val_tot_mmd += float(val_loss_mmd.item())
                if torch.isfinite(val_ce_batch):
                    val_tot_batch_ce += float(val_ce_batch.item())
                if torch.isfinite(val_ce_cancer):
                    val_tot_cancer_ce += float(val_ce_cancer.item())
                if torch.isfinite(val_raw_mmd_slide):
                    val_tot_mmd_raw_slide += float(val_raw_mmd_slide.item())
                    val_cnt_mmd_slide += 1
                if torch.isfinite(val_raw_mmd_cancer):
                    val_tot_mmd_raw_cancer += float(val_raw_mmd_cancer.item())
                    val_cnt_mmd_cancer += 1
                val_batch_losses.append(float(val_loss_total.item()))
                val_num_batches += 1

            for g in extra_val_graphs:
                vg = g.to(device)
                out_v = model(
                    vg.x,
                    vg.edge_index,
                    batch=getattr(vg, 'batch', None),
                    edge_weight=getattr(vg, 'edge_weight', None),
                    return_h=bool(cfg.get('use_mmd', False) or (cfg.get('use_wb_align', False) and cfg.get('wb_eval_loss', False))),
                )
                (
                    val_loss_total,
                    val_loss_task,
                    val_loss_batch,
                    val_loss_cancer,
                    val_loss_mmd,
                    val_ce_batch,
                    val_ce_cancer,
                    val_raw_mmd_slide,
                    val_raw_mmd_cancer,
                ) = _compute_losses(out_v, vg)
                if use_wb_align and bool(cfg.get('wb_eval_loss', False)) and wb_module is not None and support_map is not None:
                    h_v = out_v.get('h', None)
                    if h_v is not None and hasattr(vg, 'cancer_dom') and hasattr(vg, 'batch'):
                        b_v = support_map(h_v)
                        val_loss_wb, val_loss_wb_anchor, val_wb_stats = wb_module.model_loss(
                            h=h_v,
                            b=b_v,
                            cancer_dom=vg.cancer_dom[vg.batch],
                            graph_nodes=getattr(vg, 'batch', None),
                            y=getattr(vg, 'y', None),
                        )
                        if bool(val_wb_stats.get('valid', False)):
                            val_loss_total = val_loss_total + float(_wb_lambda_for_step(global_step)) * (
                                val_loss_wb + float(cfg.get('wb_anchor_weight', 0.5)) * val_loss_wb_anchor
                            )
                            val_tot_wb += float(val_loss_wb.detach().item())
                            val_tot_wb_anchor += float(val_loss_wb_anchor.detach().item())
                            val_state = _stat_to_float(val_wb_stats, 'wb_state_direction', default=float('nan'))
                            if np.isfinite(val_state):
                                val_tot_wb_state += val_state
                            val_tot_wb_active_cancers += _stat_to_float(val_wb_stats, 'wb_active_cancers', default=0.0)
                            val_tot_wb_active_spots += _stat_to_float(val_wb_stats, 'wb_active_spots', default=0.0)
                            val_cnt_wb += 1
                logits_v = out_v['logits']
                val_logits_list.append(logits_v.cpu())
                val_y_list.append(vg.y.cpu())
                m_slide = eval_logits(logits_v, vg.y)
                per_slide_acc.append(m_slide.get('accuracy', float('nan')))
                val_tot_total += float(val_loss_total.item())
                val_tot_task += float(val_loss_task.item())
                val_tot_batch += float(val_loss_batch.item())
                val_tot_cancer += float(val_loss_cancer.item())
                val_tot_mmd += float(val_loss_mmd.item())
                if torch.isfinite(val_ce_batch):
                    val_tot_batch_ce += float(val_ce_batch.item())
                if torch.isfinite(val_ce_cancer):
                    val_tot_cancer_ce += float(val_ce_cancer.item())
                if torch.isfinite(val_raw_mmd_slide):
                    val_tot_mmd_raw_slide += float(val_raw_mmd_slide.item())
                    val_cnt_mmd_slide += 1
                if torch.isfinite(val_raw_mmd_cancer):
                    val_tot_mmd_raw_cancer += float(val_raw_mmd_cancer.item())
                    val_cnt_mmd_cancer += 1
                val_batch_losses.append(float(val_loss_total.item()))
                val_num_batches += 1

        if val_logits_list and val_y_list:
            val_logits = torch.cat(val_logits_list, dim=0)
            val_y = torch.cat(val_y_list, dim=0)
            m = eval_logits(val_logits, val_y)
        else:
            m = {'auroc': float('nan'), 'auprc': float('nan'), 'accuracy': float('nan'), 'macro_f1': float('nan')}

        val_avg_total = val_tot_total / max(1, val_num_batches)
        val_avg_task = val_tot_task / max(1, val_num_batches)
        val_avg_batch = val_tot_batch / max(1, val_num_batches)
        val_avg_cancer = val_tot_cancer / max(1, val_num_batches)
        val_avg_mmd = val_tot_mmd / max(1, val_num_batches)
        val_avg_batch_ce = (val_tot_batch_ce / max(1, val_num_batches)) if cfg.get('use_domain_adv_slide', False) else float('nan')
        val_avg_cancer_ce = (val_tot_cancer_ce / max(1, val_num_batches)) if cfg.get('use_domain_adv_cancer', False) else float('nan')
        val_avg_mmd_raw_slide = (val_tot_mmd_raw_slide / val_cnt_mmd_slide) if val_cnt_mmd_slide > 0 else float('nan')
        val_avg_mmd_raw_cancer = (val_tot_mmd_raw_cancer / val_cnt_mmd_cancer) if val_cnt_mmd_cancer > 0 else float('nan')
        val_avg_wb = (val_tot_wb / val_cnt_wb) if val_cnt_wb > 0 else float('nan')
        val_avg_wb_anchor = (val_tot_wb_anchor / val_cnt_wb) if val_cnt_wb > 0 else float('nan')
        val_avg_wb_state = (val_tot_wb_state / val_cnt_wb) if val_cnt_wb > 0 else float('nan')
        val_avg_wb_active_cancers = (val_tot_wb_active_cancers / val_cnt_wb) if val_cnt_wb > 0 else float('nan')
        val_avg_wb_active_spots = (val_tot_wb_active_spots / val_cnt_wb) if val_cnt_wb > 0 else float('nan')
        val_batch_loss_variance = float(np.mean([(v - val_avg_total) ** 2 for v in val_batch_losses])) if val_batch_losses else float('nan')
        val_accuracy = float(np.nanmean(per_slide_acc)) if per_slide_acc else float('nan')
        hist['val_avg_total_loss'].append(val_avg_total)
        hist['val_avg_task_loss'].append(val_avg_task)
        hist['val_avg_batch_domain_loss'].append(val_avg_batch)
        hist['val_avg_cancer_domain_loss'].append(val_avg_cancer)
        hist['val_avg_batch_domain_ce'].append(val_avg_batch_ce)
        hist['val_avg_cancer_domain_ce'].append(val_avg_cancer_ce)
        hist['val_avg_mmd_loss'].append(val_avg_mmd)
        hist['val_avg_mmd_raw_slide'].append(val_avg_mmd_raw_slide)
        hist['val_avg_mmd_raw_cancer'].append(val_avg_mmd_raw_cancer)
        hist['val_avg_wb_loss'].append(val_avg_wb)
        hist['val_avg_wb_anchor'].append(val_avg_wb_anchor)
        hist['val_avg_wb_state_direction'].append(val_avg_wb_state)
        hist['val_avg_wb_active_cancers'].append(val_avg_wb_active_cancers)
        hist['val_avg_wb_active_spots'].append(val_avg_wb_active_spots)
        hist['val_batch_loss_variance'].append(val_batch_loss_variance)
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
        plateau_metrics = {
            'val_accuracy': val_accuracy,
            'val_avg_total_loss': val_avg_total,
            'val_macro_f1': m.get('macro_f1', float('nan')),
            'val_auroc': m.get('auroc', float('nan')),
            'val_auprc': m.get('auprc', float('nan')),
        }
        best_metric_name = str(cfg.get('best_metric', 'val_macro_f1'))
        best_metric_value = plateau_metrics.get(best_metric_name, float('nan'))
        if lr_scheduler is not None and lr_scheduler_name == 'plateau':
            monitored_metric = plateau_metrics.get(plateau_metric, float('nan'))
            if np.isfinite(monitored_metric):
                lr_scheduler.step(float(monitored_metric))
            else:
                print(
                    f"[Warn] Skip ReduceLROnPlateau.step at epoch {epoch} because "
                    f"{plateau_metric} is not finite: {monitored_metric}"
                )
        current_lr = float(opt.param_groups[0]['lr']) if opt.param_groups else float('nan')
        hist['lr'].append(current_lr)
        last_epoch = int(epoch)
        last_metrics = dict(metrics)
        if report_cb is not None:
            try:
                report_cb(epoch, metrics)
            except Exception:
                pass

        if progress_desc:
            try:
                epoch_iter.set_postfix(
                    train_loss=f'{avg_total:.3f}',
                    mmd=f'{avg_mmd:.3f}',
                    val_acc=f'{val_accuracy:.3f}',
                    lr=f'{current_lr:.2e}'
                )
            except Exception:
                pass

        if best.get('state', None) is None:
            improved = True
        elif best_metric_name == 'val_avg_total_loss':
            prev_best_metric = best.get('best_metric_value', float('inf'))
            if not np.isfinite(prev_best_metric):
                prev_best_metric = float('inf')
            improved = np.isfinite(best_metric_value) and best_metric_value < prev_best_metric
        else:
            prev_best_metric = best.get('best_metric_value', -float('inf'))
            if not np.isfinite(prev_best_metric):
                prev_best_metric = -float('inf')
            improved = np.isfinite(best_metric_value) and best_metric_value > prev_best_metric
        if improved:
            best = {
                'auroc': metrics['auroc'],
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'auprc': metrics['auprc'],
                'best_metric': best_metric_name,
                'best_metric_value': float(best_metric_value) if np.isfinite(best_metric_value) else float('nan'),
                'state': copy.deepcopy(model.state_dict()),
                'epoch': epoch,
            }
            patience = 0
        else:
            patience += 1

        if int(epoch) in capture_epoch_states and int(epoch) not in saved_epoch_states:
            saved_epoch_states[int(epoch)] = {
                'state': copy.deepcopy(model.state_dict()),
                'metrics': dict(metrics),
            }

        if (not improved) and early_stop_enabled and patience >= early_patience:
            break
    best['last_epoch'] = int(last_epoch)
    best['last_metrics'] = last_metrics if last_metrics is not None else {
        'accuracy': float('nan'),
        'macro_f1': float('nan'),
        'auroc': float('nan'),
        'auprc': float('nan'),
    }
    best['completed_full_epochs'] = bool(int(last_epoch) == int(cfg.get('epochs', 0)))
    best['epoch_states'] = saved_epoch_states
    if capture_last_state:
        best['last_state'] = copy.deepcopy(model.state_dict())
    if use_wb_align and support_map is not None and wb_module is not None:
        best['wb_support_map_last'] = copy.deepcopy(support_map.state_dict())
        best['wb_potentials_last'] = copy.deepcopy(wb_module.state_dict())
        best['wb_config'] = {
            k: cfg.get(k)
            for k in sorted(cfg.keys())
            if str(k).startswith('wb_') or str(k) in {'use_wb_align', 'lambda_wb'}
        }
    return best, hist, best['state']


def _save_loss_components_csv(hist, out_dir):
    n_epochs = len(hist.get('avg_total_loss', []))
    if n_epochs == 0:
        return None
    df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'lr': hist.get('lr', [float('nan')] * n_epochs),
        'avg_total_loss': hist.get('avg_total_loss', [float('nan')] * n_epochs),
        'avg_task_loss': hist.get('avg_task_loss', [float('nan')] * n_epochs),
        'batch_loss_variance': hist.get('batch_loss_variance', [float('nan')] * n_epochs),
        'avg_cancer_domain_ce': hist.get('avg_cancer_domain_ce', [float('nan')] * n_epochs),
        'avg_batch_domain_ce': hist.get('avg_batch_domain_ce', [float('nan')] * n_epochs),
        'avg_cancer_domain_loss': hist.get('avg_cancer_domain_loss', [float('nan')] * n_epochs),
        'avg_batch_domain_loss': hist.get('avg_batch_domain_loss', [float('nan')] * n_epochs),
        'avg_mmd_loss': hist.get('avg_mmd_loss', [float('nan')] * n_epochs),
        'avg_mmd_raw_slide': hist.get('avg_mmd_raw_slide', [float('nan')] * n_epochs),
        'avg_mmd_raw_cancer': hist.get('avg_mmd_raw_cancer', [float('nan')] * n_epochs),
        'avg_wb_loss': hist.get('avg_wb_loss', [float('nan')] * n_epochs),
        'avg_wb_potential_loss': hist.get('avg_wb_potential_loss', [float('nan')] * n_epochs),
        'avg_wb_dual_obj': hist.get('avg_wb_dual_obj', [float('nan')] * n_epochs),
        'avg_wb_euclid_pairwise': hist.get('avg_wb_euclid_pairwise', [float('nan')] * n_epochs),
        'avg_wb_anchor': hist.get('avg_wb_anchor', [float('nan')] * n_epochs),
        'avg_wb_state_direction': hist.get('avg_wb_state_direction', [float('nan')] * n_epochs),
        'avg_wb_active_cancers': hist.get('avg_wb_active_cancers', [float('nan')] * n_epochs),
        'avg_wb_active_spots': hist.get('avg_wb_active_spots', [float('nan')] * n_epochs),
        'wb_lambda': hist.get('wb_lambda', [float('nan')] * n_epochs),
        'train_batch_domain_acc': hist.get('train_batch_domain_acc', [float('nan')] * n_epochs),
        'train_cancer_domain_acc': hist.get('train_cancer_domain_acc', [float('nan')] * n_epochs),
        'train_accuracy': hist.get('train_accuracy', [float('nan')] * n_epochs),
        'val_avg_total_loss': hist.get('val_avg_total_loss', [float('nan')] * n_epochs),
        'val_avg_task_loss': hist.get('val_avg_task_loss', [float('nan')] * n_epochs),
        'val_batch_loss_variance': hist.get('val_batch_loss_variance', [float('nan')] * n_epochs),
        'val_avg_cancer_domain_ce': hist.get('val_avg_cancer_domain_ce', [float('nan')] * n_epochs),
        'val_avg_batch_domain_ce': hist.get('val_avg_batch_domain_ce', [float('nan')] * n_epochs),
        'val_avg_cancer_domain_loss': hist.get('val_avg_cancer_domain_loss', [float('nan')] * n_epochs),
        'val_avg_batch_domain_loss': hist.get('val_avg_batch_domain_loss', [float('nan')] * n_epochs),
        'val_avg_mmd_loss': hist.get('val_avg_mmd_loss', [float('nan')] * n_epochs),
        'val_avg_mmd_raw_slide': hist.get('val_avg_mmd_raw_slide', [float('nan')] * n_epochs),
        'val_avg_mmd_raw_cancer': hist.get('val_avg_mmd_raw_cancer', [float('nan')] * n_epochs),
        'val_avg_wb_loss': hist.get('val_avg_wb_loss', [float('nan')] * n_epochs),
        'val_avg_wb_anchor': hist.get('val_avg_wb_anchor', [float('nan')] * n_epochs),
        'val_avg_wb_state_direction': hist.get('val_avg_wb_state_direction', [float('nan')] * n_epochs),
        'val_avg_wb_active_cancers': hist.get('val_avg_wb_active_cancers', [float('nan')] * n_epochs),
        'val_avg_wb_active_spots': hist.get('val_avg_wb_active_spots', [float('nan')] * n_epochs),
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
        return None, None, None, None
    epochs = list(range(1, n_epochs + 1))
    train_color = '#1f3a5f'
    val_color = '#d97757'
    do_smooth = n_epochs > 100
    smooth_window = 10
    raw_alpha = 0.35
    raw_lw = 0.6
    smooth_lw = 1.4

    def _draw_series(ax, values, color, label=None):
        series = pd.Series(values, dtype='float64')
        if do_smooth:
            ax.plot(
                epochs,
                series.values,
                color=color,
                linewidth=raw_lw,
                alpha=raw_alpha,
            )
            smooth = series.rolling(window=smooth_window, min_periods=1).mean()
            ax.plot(
                epochs,
                smooth.values,
                color=color,
                linewidth=smooth_lw,
                label=label,
            )
        else:
            ax.plot(
                epochs,
                series.values,
                color=color,
                linewidth=smooth_lw,
                label=label,
            )

    def _plot_neurips(ax, values, title, val_values=None):
        _draw_series(ax, values, train_color, label='train' if val_values is not None else None)
        if val_values is not None:
            _draw_series(ax, val_values, val_color, label='val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        if val_values is not None:
            ax.legend(loc='best', fontsize=8, frameon=False)

    # 1) 学习率曲线
    fig_lr, ax_lr = plt.subplots(1, 1, figsize=(8, 4.5), sharex=True)
    lr_values = hist.get('lr', [float('nan')] * n_epochs)
    _plot_neurips(ax_lr, lr_values, 'lr')
    fig_lr.tight_layout()
    out_lr_svg = os.path.join(out_dir, 'lr.svg')
    fig_lr.savefig(out_lr_svg, format='svg', dpi=150)
    plt.close(fig_lr)

    # 2) 训练损失图（3x3）
    fig1, axes1 = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    metrics_train = [
        ('avg_total_loss', 'avg_total_loss'),
        ('avg_task_loss', 'avg_task_loss'),
        ('avg_mmd_loss', 'avg_mmd_loss'),
        ('batch_loss_variance', 'batch_loss_variance'),
        ('avg_cancer_domain_loss', 'avg_cancer_domain_loss'),
        ('avg_batch_domain_loss', 'avg_batch_domain_loss'),
        ('avg_mmd_raw_slide', 'avg_mmd_raw_slide'),
        ('avg_mmd_raw_cancer', 'avg_mmd_raw_cancer'),
        ('train_accuracy', 'train_accuracy'),
    ]
    for ax, (key, title) in zip(axes1.flatten(), metrics_train):
        values = hist.get(key, [float('nan')] * n_epochs)
        _plot_neurips(ax, values, title)
    for ax in axes1[0]:
        ax.tick_params(labelbottom=True)
    for ax in axes1[1]:
        ax.tick_params(labelbottom=True)
    fig1.tight_layout()
    out_train_svg = os.path.join(out_dir, 'train_loss.svg')
    fig1.savefig(out_train_svg, format='svg', dpi=150)
    plt.close(fig1)

    # 3) 训练/验证损失对照图（3x3）
    fig_mid, axes_mid = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    metrics_train_val = [
        ('avg_total_loss', 'val_avg_total_loss', 'avg_total_loss'),
        ('avg_task_loss', 'val_avg_task_loss', 'avg_task_loss'),
        ('avg_mmd_loss', 'val_avg_mmd_loss', 'avg_mmd_loss'),
        ('batch_loss_variance', 'val_batch_loss_variance', 'batch_loss_variance'),
        ('avg_cancer_domain_loss', 'val_avg_cancer_domain_loss', 'avg_cancer_domain_loss'),
        ('avg_batch_domain_loss', 'val_avg_batch_domain_loss', 'avg_batch_domain_loss'),
        ('avg_mmd_raw_slide', 'val_avg_mmd_raw_slide', 'avg_mmd_raw_slide'),
        ('avg_mmd_raw_cancer', 'val_avg_mmd_raw_cancer', 'avg_mmd_raw_cancer'),
        ('train_accuracy', 'val_accuracy', 'accuracy'),
    ]
    for ax, (train_key, val_key, title) in zip(axes_mid.flatten(), metrics_train_val):
        train_values = hist.get(train_key, [float('nan')] * n_epochs)
        val_values = hist.get(val_key, [float('nan')] * n_epochs)
        _plot_neurips(ax, train_values, title, val_values=val_values)
    for ax in axes_mid[0]:
        ax.tick_params(labelbottom=True)
    for ax in axes_mid[1]:
        ax.tick_params(labelbottom=True)
    fig_mid.tight_layout()
    out_train_val_loss_svg = os.path.join(out_dir, 'train_val_loss.svg')
    fig_mid.savefig(out_train_val_loss_svg, format='svg', dpi=150)
    plt.close(fig_mid)

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
    return out_lr_svg, out_train_svg, out_train_val_loss_svg, out_val_svg


def _plot_wb_train_metrics(hist, out_dir):
    n_epochs = len(hist.get('avg_total_loss', []))
    if n_epochs == 0 or 'avg_wb_loss' not in hist:
        return None
    epochs = list(range(1, n_epochs + 1))
    line_color = '#1f3a5f'
    do_smooth = n_epochs > 100
    smooth_window = 10
    raw_alpha = 0.35
    raw_lw = 0.6
    smooth_lw = 1.4

    def _plot_neurips(ax, values, title):
        series = pd.Series(values, dtype='float64')
        if do_smooth:
            ax.plot(epochs, series.values, color=line_color, linewidth=raw_lw, alpha=raw_alpha)
            smooth = series.rolling(window=smooth_window, min_periods=1).mean()
            ax.plot(epochs, smooth.values, color=line_color, linewidth=smooth_lw)
        else:
            ax.plot(epochs, series.values, color=line_color, linewidth=smooth_lw)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    metrics_wb = [
        ('avg_total_loss', 'avg_total_loss'),
        ('avg_task_loss', 'avg_task_loss'),
        ('avg_wb_loss', 'avg_wb_loss'),
        ('avg_wb_potential_loss', 'avg_wb_potential_loss'),
        ('avg_wb_dual_obj', 'avg_wb_dual_obj'),
        ('avg_wb_euclid_pairwise', 'avg_wb_euclid_pairwise'),
        ('avg_wb_anchor', 'avg_wb_anchor'),
        ('avg_wb_state_direction', 'avg_wb_state_direction'),
        ('wb_lambda', 'wb_lambda'),
        ('avg_wb_active_cancers', 'avg_wb_active_cancers'),
        ('avg_wb_active_spots', 'avg_wb_active_spots'),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(17, 12), sharex=True)
    axes_flat = axes.flatten()
    for ax, (key, title) in zip(axes_flat, metrics_wb):
        values = hist.get(key, [float('nan')] * n_epochs)
        _plot_neurips(ax, values, title)
    for ax in axes_flat[len(metrics_wb):]:
        ax.axis('off')
    for row in axes[:-1]:
        for ax in row:
            ax.tick_params(labelbottom=True)
    fig.tight_layout()
    out_svg = os.path.join(out_dir, 'wb_train_loss.svg')
    fig.savefig(out_svg, format='svg', dpi=150)
    plt.close(fig)
    return out_svg


def _count_seen_domains(graphs):
    batch_seen = {int(g.bat_dom.item()) for g in graphs if hasattr(g, 'bat_dom')}
    cancer_seen = {int(g.cancer_dom.item()) for g in graphs if hasattr(g, 'cancer_dom')}
    n_batch = len(batch_seen) if batch_seen else None
    n_cancer = len(cancer_seen) if cancer_seen else None
    return n_batch, n_cancer


def _save_wb_artifacts(best, out_dir):
    if not isinstance(best, dict) or 'wb_support_map_last' not in best or 'wb_potentials_last' not in best:
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    support_path = os.path.join(out_dir, 'wb_support_map_last.pt')
    potentials_path = os.path.join(out_dir, 'wb_potentials_last.pt')
    config_path = os.path.join(out_dir, 'wb_config.json')
    torch.save(best['wb_support_map_last'], support_path)
    torch.save(best['wb_potentials_last'], potentials_path)
    save_json(dict(best.get('wb_config', {}) or {}), config_path)
    paths.extend([support_path, potentials_path, config_path])
    return paths


def _plot_domain_diagnostics(
    hist,
    out_dir,
    n_domains_batch=None,
    n_domains_cancer=None,
    lambda_slide=0.0,
    lambda_cancer=0.0,
    smooth_window=10,
    title='Domain Adversarial Diagnostics (CE & Accuracy)',
):
    """2x2 诊断图：
    - Batch domain: unweighted CE vs log(K), Accuracy vs 1/K
    - Cancer domain: unweighted CE vs log(K), Accuracy vs 1/K

    注意：
    - 优先使用 hist['avg_*_domain_ce']（unweighted CE）。
    - 若缺失，则回退用 hist['avg_*_domain_loss'] / lambda_* 近似还原（当 lambda_*=0 或域头未启用时会显示 NaN）。
    """
    n_epochs = len(hist.get('avg_total_loss', []))
    if n_epochs == 0:
        return None
    epochs = list(range(1, n_epochs + 1))

    do_smooth = n_epochs > 100
    line_color = '#1f3a5f'
    raw_alpha = 0.35
    raw_lw = 0.6
    smooth_lw = 1.4
    chance_lw = smooth_lw

    def _series(key):
        vals = hist.get(key, [])
        if len(vals) < n_epochs:
            vals = list(vals) + [float('nan')] * (n_epochs - len(vals))
        return pd.Series(vals, dtype='float64')

    batch_loss = _series('avg_batch_domain_loss')
    cancer_loss = _series('avg_cancer_domain_loss')
    batch_ce_raw = _series('avg_batch_domain_ce')
    cancer_ce_raw = _series('avg_cancer_domain_ce')
    batch_acc = _series('train_batch_domain_acc')
    cancer_acc = _series('train_cancer_domain_acc')

    if batch_ce_raw.notna().any():
        batch_ce = batch_ce_raw
    else:
        if lambda_slide and float(lambda_slide) != 0.0:
            batch_ce = batch_loss / float(lambda_slide)
        else:
            batch_ce = pd.Series([float('nan')] * n_epochs, dtype='float64')
    if cancer_ce_raw.notna().any():
        cancer_ce = cancer_ce_raw
    else:
        if lambda_cancer and float(lambda_cancer) != 0.0:
            cancer_ce = cancer_loss / float(lambda_cancer)
        else:
            cancer_ce = pd.Series([float('nan')] * n_epochs, dtype='float64')

    # 若域头未启用（acc 为 NaN），则 CE 也显示 NaN，避免误读为 0
    batch_ce = batch_ce.where(~batch_acc.isna(), float('nan'))
    cancer_ce = cancer_ce.where(~cancer_acc.isna(), float('nan'))

    def _smooth(series):
        if not do_smooth:
            return series
        w = int(smooth_window) if smooth_window else 10
        w = max(1, w)
        return series.rolling(window=w, min_periods=1).mean()

    batch_ce_s = _smooth(batch_ce)
    cancer_ce_s = _smooth(cancer_ce)
    batch_acc_s = _smooth(batch_acc)
    cancer_acc_s = _smooth(cancer_acc)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')

    # (1) Batch CE
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('white')
    if do_smooth:
        ax1.plot(epochs, batch_ce.values, alpha=raw_alpha, linewidth=raw_lw, color=line_color, label='CE (raw)')
        ax1.plot(epochs, batch_ce_s.values, linewidth=smooth_lw, color=line_color, label=f'CE (smooth, w={smooth_window})')
    else:
        ax1.plot(epochs, batch_ce.values, linewidth=smooth_lw, color=line_color, label='CE')
    if n_domains_batch is not None and int(n_domains_batch) > 0:
        ax1.axhline(float(np.log(int(n_domains_batch))), color=line_color, linestyle='--', linewidth=chance_lw, label=f'Chance (train, log({int(n_domains_batch)}))')
    ax1.set_title('Batch Domain CE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy (unweighted)')
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.legend(fontsize=9)

    # (2) Batch Acc
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('white')
    if do_smooth:
        ax2.plot(epochs, batch_acc.values, alpha=raw_alpha, linewidth=raw_lw, color=line_color, label='Accuracy (raw)')
        ax2.plot(epochs, batch_acc_s.values, linewidth=smooth_lw, color=line_color, label=f'Accuracy (smooth, w={smooth_window})')
    else:
        ax2.plot(epochs, batch_acc.values, linewidth=smooth_lw, color=line_color, label='Accuracy')
    if n_domains_batch is not None and int(n_domains_batch) > 0:
        ax2.axhline(1.0 / float(int(n_domains_batch)), color=line_color, linestyle='--', linewidth=chance_lw, label=f'Chance accuracy (train, 1/{int(n_domains_batch)})')
    ax2.set_title('Batch Domain Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.legend(fontsize=9)

    # (3) Cancer CE
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('white')
    if do_smooth:
        ax3.plot(epochs, cancer_ce.values, alpha=raw_alpha, linewidth=raw_lw, color=line_color, label='CE (raw)')
        ax3.plot(epochs, cancer_ce_s.values, linewidth=smooth_lw, color=line_color, label=f'CE (smooth, w={smooth_window})')
    else:
        ax3.plot(epochs, cancer_ce.values, linewidth=smooth_lw, color=line_color, label='CE')
    if n_domains_cancer is not None and int(n_domains_cancer) > 0:
        ax3.axhline(float(np.log(int(n_domains_cancer))), color=line_color, linestyle='--', linewidth=chance_lw, label=f'Chance (train, log({int(n_domains_cancer)}))')
    ax3.set_title('Cancer Domain CE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Cross-Entropy (unweighted)')
    ax3.spines['top'].set_visible(True)
    ax3.spines['right'].set_visible(True)
    ax3.legend(fontsize=9)

    # (4) Cancer Acc
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('white')
    if do_smooth:
        ax4.plot(epochs, cancer_acc.values, alpha=raw_alpha, linewidth=raw_lw, color=line_color, label='Accuracy (raw)')
        ax4.plot(epochs, cancer_acc_s.values, linewidth=smooth_lw, color=line_color, label=f'Accuracy (smooth, w={smooth_window})')
    else:
        ax4.plot(epochs, cancer_acc.values, linewidth=smooth_lw, color=line_color, label='Accuracy')
    if n_domains_cancer is not None and int(n_domains_cancer) > 0:
        ax4.axhline(1.0 / float(int(n_domains_cancer)), color=line_color, linestyle='--', linewidth=chance_lw, label=f'Chance accuracy (train, 1/{int(n_domains_cancer)})')
    ax4.set_title('Cancer Domain Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.spines['top'].set_visible(True)
    ax4.spines['right'].set_visible(True)
    ax4.legend(fontsize=9)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out_svg = os.path.join(out_dir, 'train_domain_diagnostics.svg')
    fig.savefig(out_svg, format='svg', dpi=150)
    plt.close(fig)
    return out_svg


def run_single_training(args, cfg, device):
    """原始的单次训练逻辑（现支持双域对抗）"""
    
    train_graphs, val_graphs, in_dim, n_domains_batch, n_domains_cancer = prepare_graphs(args, cfg, save_preprocessor_dir=args.artifacts_dir)
    train_seen_n_domains_batch, train_seen_n_domains_cancer = _count_seen_domains(train_graphs)

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
        capture_last_state=bool(getattr(args, 'save_last', False)),
        capture_epoch_states=getattr(args, '_save_epoch_checkpoints', []),
    )

    # 保存模型：默认保存最优到 model.pt；若 --save_last 且实际跑满 epochs，则用最后一个 epoch 覆盖 model.pt，并额外保存 best 到 model_best.pt
    model = STOnco_Classifier(
        in_dim=in_dim,
        hidden=cfg['GNN_hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
        clf_hidden=cfg.get('clf_hidden', [256, 128, 64]),
        domain_hidden=int(cfg.get('dom_hidden', 64)),
        use_domain_adv_slide=cfg.get('use_domain_adv_slide', False), n_domains_slide=n_domains_batch,
        use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
    )
    save_last = bool(getattr(args, 'save_last', False))
    completed_full_epochs = bool(best.get('completed_full_epochs', False))
    last_state = best.get('last_state', None)

    model.load_state_dict(best_state)
    if save_last:
        save_model(model, args.artifacts_dir, filename='model_best.pt')
        if completed_full_epochs and last_state is not None:
            model.load_state_dict(last_state)
            save_model(model, args.artifacts_dir)  # model.pt
            saved_checkpoint = 'last'
        else:
            save_model(model, args.artifacts_dir)  # model.pt
            saved_checkpoint = 'best'
    else:
        save_model(model, args.artifacts_dir)  # model.pt
        saved_checkpoint = 'best'
    wb_artifact_paths = _save_wb_artifacts(best, args.artifacts_dir) if bool(cfg.get('use_wb_align', False)) else []
    for wb_path in wb_artifact_paths:
        print('Saved WB artifact to', wb_path)
    train_ids = list(getattr(args, '_train_ids', []))
    val_ids = list(getattr(args, '_val_ids', []))
    requested_save_epochs = list(getattr(args, '_save_epoch_checkpoints', []))
    extra_saved_epochs = _save_extra_epoch_checkpoints(
        model,
        best.get('epoch_states', {}),
        cfg,
        args.artifacts_dir,
        extra_meta={'train_ids': train_ids, 'val_ids': val_ids},
        hist=hist,
        save_train_curves=bool(getattr(args, 'save_train_curves', 1)),
        save_loss_components=bool(getattr(args, 'save_loss_components', 0)),
        domain_plot_kwargs={
            'n_domains_batch': train_seen_n_domains_batch,
            'n_domains_cancer': train_seen_n_domains_cancer,
            'lambda_slide': cfg.get('lambda_slide', 0.0),
            'lambda_cancer': cfg.get('lambda_cancer', 0.0),
        },
        artifact_source_dir=args.artifacts_dir,
    )
    missing_save_epochs = sorted(set(requested_save_epochs) - set(extra_saved_epochs))
    if len(missing_save_epochs) > 0:
        print(f"[Warn] Requested save_epoch_checkpoints were not reached: {missing_save_epochs}")
    meta = {
        'cfg': cfg,
        'best_epoch': best['epoch'],
        'best_metric': best.get('best_metric', cfg.get('best_metric', 'val_macro_f1')),
        'best_metric_value': best.get('best_metric_value', float('nan')),
        'last_epoch': int(best.get('last_epoch', best.get('epoch', -1))),
        'completed_full_epochs': bool(completed_full_epochs),
        'saved_checkpoint': saved_checkpoint,
        'saved_epoch_checkpoints': extra_saved_epochs,
        'requested_save_epoch_checkpoints': requested_save_epochs,
        'val_ids': val_ids,
        'train_ids': train_ids,
        'metrics': {
            'auroc': best.get('auroc', float('nan')),
            'auprc': best.get('auprc', float('nan')),
            'accuracy': best.get('accuracy', float('nan')),
            'macro_f1': best.get('macro_f1', float('nan')),
        }
    }
    # 追加：最后一个epoch的指标（与 saved_checkpoint 对齐）
    meta['last_metrics'] = dict(best.get('last_metrics', {}))
    save_json(meta, os.path.join(args.artifacts_dir, 'meta.json'))
    print('Saved artifacts to', args.artifacts_dir)

    # 保存训练与验证曲线
    try:
        os.makedirs(args.artifacts_dir, exist_ok=True)
        if getattr(args, 'save_train_curves', 1):
            out_lr_svg, out_train_svg, out_train_val_loss_svg, out_val_svg = _plot_train_metrics(hist, args.artifacts_dir)
            if out_lr_svg:
                print('Saved learning-rate figure to', out_lr_svg)
            if out_train_svg:
                print('Saved training loss figure to', out_train_svg)
            if out_train_val_loss_svg:
                print('Saved training/validation loss figure to', out_train_val_loss_svg)
            if out_val_svg:
                print('Saved validation metrics figure to', out_val_svg)
            out_dom_svg = _plot_domain_diagnostics(
                hist,
                args.artifacts_dir,
                n_domains_batch=train_seen_n_domains_batch,
                n_domains_cancer=train_seen_n_domains_cancer,
                lambda_slide=cfg.get('lambda_slide', 0.0),
                lambda_cancer=cfg.get('lambda_cancer', 0.0),
            )
            if out_dom_svg:
                print('Saved domain diagnostics figure to', out_dom_svg)
            if bool(cfg.get('use_wb_align', False)):
                out_wb_svg = _plot_wb_train_metrics(hist, args.artifacts_dir)
                if out_wb_svg:
                    print('Saved WB diagnostics figure to', out_wb_svg)

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
    files = set(data.files)
    Xs = list(data['Xs']); ys = list(data['ys']); xys = list(data['xys']); slide_ids = list(data['slide_ids'])
    gene_names = list(data['gene_names'])

    use_image_features = bool(cfg.get('use_image_features', False))
    X_imgs = None
    img_masks = None
    img_feature_names = None
    if use_image_features:
        required = {'X_imgs', 'img_masks', 'img_feature_names'}
        missing = sorted(list(required - files))
        if missing:
            raise ValueError(f"use_image_features=1 requires keys {sorted(required)} in train_npz, missing: {missing}")
        X_imgs = list(data['X_imgs'])
        img_masks = list(data['img_masks'])
        img_feature_names = list(data['img_feature_names'])
        if not (len(X_imgs) == len(img_masks) == len(Xs)):
            raise ValueError(
                f'Length mismatch in train_npz: len(Xs)={len(Xs)}, len(X_imgs)={len(X_imgs)}, len(img_masks)={len(img_masks)}'
            )

    slides = []
    for i, (X, y, xy, sid) in enumerate(zip(Xs, ys, xys, slide_ids)):
        item = {'X': X, 'y': y, 'xy': xy, 'slide_id': sid}
        if use_image_features:
            item['X_img'] = X_imgs[i]
            item['img_mask'] = img_masks[i]
        slides.append(item)

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

    img_pp = None
    if use_image_features:
        img_pp = ImagePreprocessor(img_use_pca=cfg.get('img_use_pca', True), img_pca_dim=cfg.get('img_pca_dim', 256))
        img_pp.fit([s['X_img'] for s in slides], [s['img_mask'] for s in slides])

    pyg_graphs = []
    for s in slides:
        Xp_gene = pp.transform(s['X'], gene_names)
        if use_image_features:
            Xp_img = img_pp.transform(s['X_img'], s['img_mask'])
            data_g = assemble_pyg(Xp_gene, s['xy'], s['y'], cfg, Xp_img=Xp_img, img_mask=s['img_mask'])
        else:
            data_g = assemble_pyg(Xp_gene, s['xy'], s['y'], cfg)
        data_g.slide_id = str(s['slide_id'])
        batch_id = _resolve_batch_id(str(s['slide_id']), id2batch, batch_fallback_cache)
        data_g.bat_dom = torch.tensor(batch_to_idx[batch_id], dtype=torch.long)
        ctype = id2type.get(str(s['slide_id']))
        data_g.cancer_dom = torch.tensor(cancer_to_idx[ctype], dtype=torch.long)
        pyg_graphs.append(data_g)

    in_dim = pyg_graphs[0].x.shape[1]

    n_domains_batch = len(batch_to_idx) if cfg.get('use_domain_adv_slide', False) else None
    n_domains_cancer = len(cancer_to_idx) if (cfg.get('use_domain_adv_cancer', False) or cfg.get('use_wb_align', False)) else None

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
            if use_image_features:
                img_pp.save(fold_dir, img_feature_names=img_feature_names)
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
            capture_last_state=bool(getattr(args, 'save_last', False)),
            capture_epoch_states=getattr(args, '_save_epoch_checkpoints', []),
        )
        train_seen_n_domains_batch, train_seen_n_domains_cancer = _count_seen_domains(train_graphs)
        if bool(cfg.get('use_wb_align', False)):
            for wb_path in _save_wb_artifacts(best, fold_dir):
                print(f'[KFold] Saved WB artifact to {wb_path}')

        if getattr(args, 'save_train_curves', 1):
            try:
                out_lr_svg, out_train_svg, out_train_val_loss_svg, out_val_svg = _plot_train_metrics(hist, fold_dir)
                if out_lr_svg:
                    print(f'[KFold] Saved learning-rate figure to {out_lr_svg}')
                if out_train_svg:
                    print(f'[KFold] Saved training loss figure to {out_train_svg}')
                if out_train_val_loss_svg:
                    print(f'[KFold] Saved training/validation loss figure to {out_train_val_loss_svg}')
                if out_val_svg:
                    print(f'[KFold] Saved validation metrics figure to {out_val_svg}')
                out_dom_svg = _plot_domain_diagnostics(
                    hist,
                    fold_dir,
                    n_domains_batch=train_seen_n_domains_batch,
                    n_domains_cancer=train_seen_n_domains_cancer,
                    lambda_slide=cfg.get('lambda_slide', 0.0),
                    lambda_cancer=cfg.get('lambda_cancer', 0.0),
                )
                if out_dom_svg:
                    print(f'[KFold] Saved domain diagnostics figure to {out_dom_svg}')
                if bool(cfg.get('use_wb_align', False)):
                    out_wb_svg = _plot_wb_train_metrics(hist, fold_dir)
                    if out_wb_svg:
                        print(f'[KFold] Saved WB diagnostics figure to {out_wb_svg}')
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

        # 保存模型：默认保存最优到 model.pt；若 --save_last 且实际跑满 epochs，则用最后一个 epoch 覆盖 model.pt，并额外保存 best 到 model_best.pt
        model = STOnco_Classifier(
            in_dim=in_dim,
            hidden=cfg['GNN_hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
            clf_hidden=cfg.get('clf_hidden', [256, 128, 64]),
            domain_hidden=int(cfg.get('dom_hidden', 64)),
            use_domain_adv_slide=cfg.get('use_domain_adv_slide', False), n_domains_slide=n_domains_batch,
            use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
        )
        save_last = bool(getattr(args, 'save_last', False))
        completed_full_epochs = bool(best.get('completed_full_epochs', False))
        last_state = best.get('last_state', None)

        model.load_state_dict(best_state)
        if save_last:
            save_model(model, fold_dir, filename='model_best.pt')
            if completed_full_epochs and last_state is not None:
                model.load_state_dict(last_state)
                save_model(model, fold_dir)  # model.pt
                saved_checkpoint = 'last'
            else:
                save_model(model, fold_dir)  # model.pt
                saved_checkpoint = 'best'
        else:
            save_model(model, fold_dir)  # model.pt
            saved_checkpoint = 'best'
        model_path = os.path.join(fold_dir, 'model.pt')
        if os.path.exists(model_path):
            print(f"[KFold] Saved model to {model_path}")
        else:
            print(f"[KFold] Saved model to {fold_dir} (model.pt)")
        requested_save_epochs = list(getattr(args, '_save_epoch_checkpoints', []))
        extra_saved_epochs = _save_extra_epoch_checkpoints(
            model,
            best.get('epoch_states', {}),
            cfg,
            fold_dir,
            extra_meta={'train_ids': train_ids, 'val_ids': val_ids},
            hist=hist,
            save_train_curves=bool(getattr(args, 'save_train_curves', 1)),
            save_loss_components=bool(getattr(args, 'save_loss_components', 0)),
            domain_plot_kwargs={
                'n_domains_batch': train_seen_n_domains_batch,
                'n_domains_cancer': train_seen_n_domains_cancer,
                'lambda_slide': cfg.get('lambda_slide', 0.0),
                'lambda_cancer': cfg.get('lambda_cancer', 0.0),
            },
            artifact_source_dir=fold_dir,
        )
        missing_save_epochs = sorted(set(requested_save_epochs) - set(extra_saved_epochs))
        if len(missing_save_epochs) > 0:
            print(f"[KFold] Warning: requested save_epoch_checkpoints were not reached: {missing_save_epochs}")
        meta = {
            'cfg': cfg,
            'best_epoch': best['epoch'],
            'best_metric': best.get('best_metric', cfg.get('best_metric', 'val_macro_f1')),
            'best_metric_value': best.get('best_metric_value', float('nan')),
            'last_epoch': int(best.get('last_epoch', best.get('epoch', -1))),
            'completed_full_epochs': bool(completed_full_epochs),
            'saved_checkpoint': saved_checkpoint,
            'saved_epoch_checkpoints': extra_saved_epochs,
            'requested_save_epoch_checkpoints': requested_save_epochs,
            'val_ids': val_ids,
            'train_ids': train_ids,
            'metrics': {
                'auroc': best.get('auroc', float('nan')),
                'auprc': best.get('auprc', float('nan')),
                'accuracy': best.get('accuracy', float('nan')),
                'macro_f1': best.get('macro_f1', float('nan')),
            }
        }
        meta['last_metrics'] = dict(best.get('last_metrics', {}))
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
    _NAN = float('nan')

    def _safe_metrics(meta, key):
        try:
            return dict(meta.get(key, {}) or {})
        except Exception:
            return {}

    def _loco_collect_completed(base_dir: str):
        rows = []
        per_slide_dfs = []
        for name in sorted(os.listdir(base_dir)):
            d = os.path.join(base_dir, name)
            if not os.path.isdir(d):
                continue
            meta_path = os.path.join(d, 'meta.json')
            if not os.path.exists(meta_path):
                continue
            try:
                meta = load_json(meta_path)
            except Exception:
                continue
            metrics_best = _safe_metrics(meta, 'metrics')
            metrics_last = _safe_metrics(meta, 'last_metrics')
            saved_checkpoint = str(meta.get('saved_checkpoint', 'best'))
            best_epoch = int(meta.get('best_epoch', -1))
            last_epoch = int(meta.get('last_epoch', best_epoch))
            saved_epoch = last_epoch if saved_checkpoint == 'last' else best_epoch
            saved_metrics = dict(metrics_best)
            if saved_checkpoint == 'last' and len(metrics_last) > 0:
                saved_metrics.update(metrics_last)
            rows.append({
                'cancer_type': name,
                'saved_checkpoint': saved_checkpoint,
                'completed_full_epochs': bool(meta.get('completed_full_epochs', False)),
                'best_epoch': best_epoch,
                'last_epoch': last_epoch,
                'saved_epoch': int(saved_epoch),
                # 兼容旧列：默认仍汇总 best（meta['metrics']）
                'auroc': metrics_best.get('auroc', _NAN),
                'auprc': metrics_best.get('auprc', _NAN),
                'accuracy': metrics_best.get('accuracy', _NAN),
                'macro_f1': metrics_best.get('macro_f1', _NAN),
                # 新增：last 与 saved（与 model.pt 对齐）
                'last_auroc': metrics_last.get('auroc', _NAN),
                'last_auprc': metrics_last.get('auprc', _NAN),
                'last_accuracy': metrics_last.get('accuracy', _NAN),
                'last_macro_f1': metrics_last.get('macro_f1', _NAN),
                'saved_auroc': saved_metrics.get('auroc', _NAN),
                'saved_auprc': saved_metrics.get('auprc', _NAN),
                'saved_accuracy': saved_metrics.get('accuracy', _NAN),
                'saved_macro_f1': saved_metrics.get('macro_f1', _NAN),
                'n_train': int(len(meta.get('train_ids', []) or [])),
                'n_val': int(len(meta.get('val_ids', []) or [])),
            })
            ps_path = os.path.join(d, 'per_slide.csv')
            if os.path.exists(ps_path):
                try:
                    per_slide_dfs.append(pd.read_csv(ps_path))
                except Exception:
                    pass
        return rows, per_slide_dfs

    def _loco_write_summaries(base_dir: str):
        results_rows, per_slide_dfs = _loco_collect_completed(base_dir)

        # 汇总CSV（整体节点合并后的指标，不混入样本均值）
        try:
            if len(results_rows) > 0:
                df = pd.DataFrame(results_rows)
                summary_path = os.path.join(base_dir, 'loco_summary.csv')
                df.to_csv(summary_path, index=False)
        except Exception as e:
            print(f"Warning: failed to write LOCO summary CSV: {e}")

        # 顶层逐样本与“样本均值”汇总（若存在 per_slide.csv）
        try:
            if len(per_slide_dfs) > 0:
                df_ps = pd.concat(per_slide_dfs, ignore_index=True)
                per_slide_path = os.path.join(base_dir, 'loco_per_slide.csv')
                df_ps.to_csv(per_slide_path, index=False)

                metrics = ['auroc', 'auprc', 'accuracy', 'macro_f1']
                summary_rows = []
                for ct, sub in df_ps.groupby('cancer_type'):
                    row = {'cancer_type': ct, 'n_val_slides': int(len(sub))}
                    for m in metrics:
                        row[f'mean_{m}_slide'] = float(sub[m].mean()) if m in sub else float('nan')
                        row[f'std_{m}_slide'] = float(sub[m].std()) if m in sub else float('nan')
                    summary_rows.append(row)
                all_row = {'cancer_type': 'ALL', 'n_val_slides': int(len(df_ps))}
                for m in metrics:
                    all_row[f'mean_{m}_slide'] = float(df_ps[m].mean()) if m in df_ps else float('nan')
                    all_row[f'std_{m}_slide'] = float(df_ps[m].std()) if m in df_ps else float('nan')
                summary_rows.append(all_row)
                df_ps_sum = pd.DataFrame(summary_rows)
                per_slide_sum_path = os.path.join(base_dir, 'loco_per_slide_summary.csv')
                df_ps_sum.to_csv(per_slide_sum_path, index=False)
        except Exception as e:
            print(f"Warning: failed to write per-slide CSVs: {e}")

    data = np.load(args.train_npz, allow_pickle=True)
    files = set(data.files)
    Xs = list(data['Xs']); ys = list(data['ys']); xys = list(data['xys']); slide_ids = list(map(str, list(data['slide_ids'])))
    gene_names = list(data['gene_names'])

    use_image_features = bool(cfg.get('use_image_features', False))
    X_imgs = None
    img_masks = None
    img_feature_names = None
    if use_image_features:
        required = {'X_imgs', 'img_masks', 'img_feature_names'}
        missing = sorted(list(required - files))
        if missing:
            raise ValueError(f"use_image_features=1 requires keys {sorted(required)} in train_npz, missing: {missing}")
        X_imgs = list(data['X_imgs'])
        img_masks = list(data['img_masks'])
        img_feature_names = list(data['img_feature_names'])
        if not (len(X_imgs) == len(img_masks) == len(Xs)):
            raise ValueError(
                f'Length mismatch in train_npz: len(Xs)={len(Xs)}, len(X_imgs)={len(X_imgs)}, len(img_masks)={len(img_masks)}'
            )

    id2type, type2present = _build_type_to_present_ids(slide_ids)
    _, _, id2batch, _ = _load_cancer_labels()
    cancer_types = sorted(type2present.keys())

    parent_dir = os.path.abspath(os.path.join(args.artifacts_dir, os.pardir))
    base_loco_dir = os.path.join(parent_dir, 'loco_eval')
    os.makedirs(base_loco_dir, exist_ok=True)

    results = []
    # 新增：跨癌种的逐样本记录（顶层汇总输出；仅用于本次进程内统计）
    per_slide_rows = []

    # 将原始slides打包便于按索引选择
    slides_all = []
    for i, (X, y, xy, sid) in enumerate(zip(Xs, ys, xys, slide_ids)):
        item = {'X': X, 'y': y, 'xy': xy, 'slide_id': sid}
        if use_image_features:
            item['X_img'] = X_imgs[i]
            item['img_mask'] = img_masks[i]
        slides_all.append(item)

    for ct in tqdm(cancer_types, desc='LOCO', total=len(cancer_types)):
        val_ids = sorted(type2present[ct])
        train_ids = sorted([sid for sid in slide_ids if sid not in set(val_ids)])
        if not train_ids or not val_ids:
            print(f"[LOCO] Skip cancer_type={ct} due to empty train/val partition.")
            continue
        loco_dir = os.path.join(base_loco_dir, ct)
        os.makedirs(loco_dir, exist_ok=True)

        # 断点续跑：若已存在完成标记（meta.json + model.pt 且 cfg 匹配），则跳过该癌种
        if bool(getattr(args, 'loco_force_rerun', False)) is False and bool(getattr(args, 'loco_resume', False)) is True:
            meta_path = os.path.join(loco_dir, 'meta.json')
            model_path = os.path.join(loco_dir, 'model.pt')
            if os.path.exists(meta_path) and os.path.exists(model_path):
                try:
                    meta_prev = load_json(meta_path)
                    if dict(meta_prev.get('cfg', {})) == dict(cfg):
                        metrics_prev = _safe_metrics(meta_prev, 'metrics')
                        last_metrics_prev = _safe_metrics(meta_prev, 'last_metrics')
                        saved_checkpoint_prev = str(meta_prev.get('saved_checkpoint', 'best'))
                        best_epoch_prev = int(meta_prev.get('best_epoch', -1))
                        last_epoch_prev = int(meta_prev.get('last_epoch', best_epoch_prev))
                        saved_epoch_prev = last_epoch_prev if saved_checkpoint_prev == 'last' else best_epoch_prev
                        saved_metrics_prev = dict(metrics_prev)
                        if saved_checkpoint_prev == 'last' and len(last_metrics_prev) > 0:
                            saved_metrics_prev.update(last_metrics_prev)
                        results.append({
                            'cancer_type': ct,
                            'saved_checkpoint': saved_checkpoint_prev,
                            'completed_full_epochs': bool(meta_prev.get('completed_full_epochs', False)),
                            'best_epoch': best_epoch_prev,
                            'last_epoch': last_epoch_prev,
                            'saved_epoch': int(saved_epoch_prev),
                            'auroc': metrics_prev.get('auroc', _NAN),
                            'auprc': metrics_prev.get('auprc', _NAN),
                            'accuracy': metrics_prev.get('accuracy', _NAN),
                            'macro_f1': metrics_prev.get('macro_f1', _NAN),
                            'last_auroc': last_metrics_prev.get('auroc', _NAN),
                            'last_auprc': last_metrics_prev.get('auprc', _NAN),
                            'last_accuracy': last_metrics_prev.get('accuracy', _NAN),
                            'last_macro_f1': last_metrics_prev.get('macro_f1', _NAN),
                            'saved_auroc': saved_metrics_prev.get('auroc', _NAN),
                            'saved_auprc': saved_metrics_prev.get('auprc', _NAN),
                            'saved_accuracy': saved_metrics_prev.get('accuracy', _NAN),
                            'saved_macro_f1': saved_metrics_prev.get('macro_f1', _NAN),
                            'n_train': int(len(meta_prev.get('train_ids', []) or [])),
                            'n_val': int(len(meta_prev.get('val_ids', []) or [])),
                        })
                        print(f'[LOCO] Resume: skip cancer_type={ct} (found existing meta/model with matching cfg)')
                        _loco_write_summaries(base_loco_dir)
                        continue
                    else:
                        print(f'[LOCO] Resume: cfg mismatch for cancer_type={ct}, will re-train and overwrite')
                except Exception as e:
                    print(f'[LOCO] Resume: failed to read meta for cancer_type={ct}, will re-train. err={e}')

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

        img_pp = None
        if use_image_features:
            img_pp = ImagePreprocessor(img_use_pca=cfg.get('img_use_pca', True), img_pca_dim=cfg.get('img_pca_dim', 256))
            img_pp.fit([s['X_img'] for s in slides_train], [s['img_mask'] for s in slides_train])
            try:
                img_pp.save(loco_dir, img_feature_names=img_feature_names)
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
                Xp_gene = pp.transform(s['X'], gene_names)
                if use_image_features:
                    Xp_img = img_pp.transform(s['X_img'], s['img_mask'])
                    g = assemble_pyg(Xp_gene, s['xy'], s['y'], cfg, Xp_img=Xp_img, img_mask=s['img_mask'])
                else:
                    g = assemble_pyg(Xp_gene, s['xy'], s['y'], cfg)
                g.slide_id = str(s['slide_id'])
                batch_id = _resolve_batch_id(str(s['slide_id']), id2batch, batch_fallback_cache)
                g.bat_dom = torch.tensor(batch_to_idx[batch_id], dtype=torch.long)
                g.cancer_dom = torch.tensor(cancer_to_idx[id2type[str(s['slide_id'])]], dtype=torch.long)
                graphs.append(g)
            return graphs

        train_graphs = build_graph_list(train_ids)
        val_graphs = build_graph_list(val_ids)

        in_dim = train_graphs[0].x.shape[1]
        n_domains_batch = len(batch_to_idx) if cfg.get('use_domain_adv_slide', False) else None
        n_domains_cancer = len(cancer_to_idx) if (cfg.get('use_domain_adv_cancer', False) or cfg.get('use_wb_align', False)) else None

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
            capture_last_state=bool(getattr(args, 'save_last', False)),
            capture_epoch_states=getattr(args, '_save_epoch_checkpoints', []),
        )
        train_seen_n_domains_batch, train_seen_n_domains_cancer = _count_seen_domains(train_graphs)
        if bool(cfg.get('use_wb_align', False)):
            for wb_path in _save_wb_artifacts(best, loco_dir):
                print(f'[LOCO] Saved WB artifact to {wb_path}')

        if getattr(args, 'save_train_curves', 1):
            try:
                out_lr_svg, out_train_svg, out_train_val_loss_svg, out_val_svg = _plot_train_metrics(hist, loco_dir)
                if out_lr_svg:
                    print(f'[LOCO] Saved learning-rate figure to {out_lr_svg}')
                if out_train_svg:
                    print(f'[LOCO] Saved training loss figure to {out_train_svg}')
                if out_train_val_loss_svg:
                    print(f'[LOCO] Saved training/validation loss figure to {out_train_val_loss_svg}')
                if out_val_svg:
                    print(f'[LOCO] Saved validation metrics figure to {out_val_svg}')
                out_dom_svg = _plot_domain_diagnostics(
                    hist,
                    loco_dir,
                    n_domains_batch=train_seen_n_domains_batch,
                    n_domains_cancer=train_seen_n_domains_cancer,
                    lambda_slide=cfg.get('lambda_slide', 0.0),
                    lambda_cancer=cfg.get('lambda_cancer', 0.0),
                )
                if out_dom_svg:
                    print(f'[LOCO] Saved domain diagnostics figure to {out_dom_svg}')
                if bool(cfg.get('use_wb_align', False)):
                    out_wb_svg = _plot_wb_train_metrics(hist, loco_dir)
                    if out_wb_svg:
                        print(f'[LOCO] Saved WB diagnostics figure to {out_wb_svg}')
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

        # 保存模型：默认保存最优到 model.pt；若 --save_last 且实际跑满 epochs，则用最后一个 epoch 覆盖 model.pt，并额外保存 best 到 model_best.pt
        model = STOnco_Classifier(
            in_dim=in_dim,
            hidden=cfg['GNN_hidden'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], model=cfg['model'], heads=cfg['heads'],
            clf_hidden=cfg.get('clf_hidden', [256, 128, 64]),
            domain_hidden=int(cfg.get('dom_hidden', 64)),
            use_domain_adv_slide=cfg.get('use_domain_adv_slide', False), n_domains_slide=n_domains_batch,
            use_domain_adv_cancer=cfg.get('use_domain_adv_cancer', False), n_domains_cancer=n_domains_cancer
        )
        save_last = bool(getattr(args, 'save_last', False))
        completed_full_epochs = bool(best.get('completed_full_epochs', False))
        last_state = best.get('last_state', None)

        model.load_state_dict(best_state)
        if save_last:
            save_model(model, loco_dir, filename='model_best.pt')
            if completed_full_epochs and last_state is not None:
                model.load_state_dict(last_state)
                save_model(model, loco_dir)  # model.pt
                saved_checkpoint = 'last'
            else:
                save_model(model, loco_dir)  # model.pt
                saved_checkpoint = 'best'
        else:
            save_model(model, loco_dir)  # model.pt
            saved_checkpoint = 'best'
        requested_save_epochs = list(getattr(args, '_save_epoch_checkpoints', []))
        extra_saved_epochs = _save_extra_epoch_checkpoints(
            model,
            best.get('epoch_states', {}),
            cfg,
            loco_dir,
            extra_meta={'train_npz': str(args.train_npz), 'train_ids': train_ids, 'val_ids': val_ids},
            hist=hist,
            save_train_curves=bool(getattr(args, 'save_train_curves', 1)),
            save_loss_components=bool(getattr(args, 'save_loss_components', 0)),
            domain_plot_kwargs={
                'n_domains_batch': train_seen_n_domains_batch,
                'n_domains_cancer': train_seen_n_domains_cancer,
                'lambda_slide': cfg.get('lambda_slide', 0.0),
                'lambda_cancer': cfg.get('lambda_cancer', 0.0),
            },
            artifact_source_dir=loco_dir,
        )
        missing_save_epochs = sorted(set(requested_save_epochs) - set(extra_saved_epochs))
        if len(missing_save_epochs) > 0:
            print(f"[LOCO] Warning: requested save_epoch_checkpoints were not reached: {missing_save_epochs}")
        meta = {
            'cfg': cfg,
            'train_npz': str(args.train_npz),
            'best_epoch': best['epoch'],
            'best_metric': best.get('best_metric', cfg.get('best_metric', 'val_macro_f1')),
            'best_metric_value': best.get('best_metric_value', float('nan')),
            'last_epoch': int(best.get('last_epoch', best.get('epoch', -1))),
            'completed_full_epochs': bool(completed_full_epochs),
            'saved_checkpoint': saved_checkpoint,
            'saved_epoch_checkpoints': extra_saved_epochs,
            'requested_save_epoch_checkpoints': requested_save_epochs,
            'val_ids': val_ids,
            'train_ids': train_ids,
            'metrics': {
                'auroc': best.get('auroc', float('nan')),
                'auprc': best.get('auprc', float('nan')),
                'accuracy': best.get('accuracy', float('nan')),
                'macro_f1': best.get('macro_f1', float('nan')),
            }
        }
        meta['last_metrics'] = dict(best.get('last_metrics', {}))
        save_json(meta, os.path.join(loco_dir, 'meta.json'))

        best_epoch_val = int(best.get('epoch', -1))
        last_epoch_val = int(best.get('last_epoch', best_epoch_val))
        saved_epoch_val = last_epoch_val if saved_checkpoint == 'last' else best_epoch_val

        # 新增：逐样本（逐 slide）评估：每癌种先落盘到 loco_dir/per_slide.csv，顶层汇总可随时重建
        try:
            model = model.to(device)
            model.eval()
            per_slide_ct_rows = []
            with torch.no_grad():
                for g in val_graphs:
                    vg = g.to(device)
                    out_v = model(
                        vg.x,
                        vg.edge_index,
                        batch=getattr(vg, 'batch', None),
                        edge_weight=getattr(vg, 'edge_weight', None),
                    )
                    logits_v = out_v['logits']
                    m_slide = eval_logits(logits_v, vg.y)
                    y_np = vg.y.detach().cpu().numpy()
                    n_nodes = int(y_np.shape[0])
                    n_pos = int((y_np == 1).sum())
                    n_neg = int((y_np == 0).sum())
                    pos_rate = float(n_pos / n_nodes) if n_nodes > 0 else float('nan')
                    row = {
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
                        'best_epoch': best_epoch_val,
                        'last_epoch': last_epoch_val,
                        'saved_checkpoint': saved_checkpoint,
                        'saved_epoch': int(saved_epoch_val),
                    }
                    per_slide_rows.append(row)
                    per_slide_ct_rows.append(row)
            if len(per_slide_ct_rows) > 0:
                try:
                    pd.DataFrame(per_slide_ct_rows).to_csv(os.path.join(loco_dir, 'per_slide.csv'), index=False)
                except Exception as e:
                    print(f"[LOCO] Warning: failed to write per_slide.csv for cancer_type={ct}: {e}")
        except Exception as e:
            print(f"[LOCO] Warning: per-slide evaluation failed for cancer_type={ct}: {e}")

        results.append({
            'cancer_type': ct,
            'saved_checkpoint': saved_checkpoint,
            'completed_full_epochs': bool(completed_full_epochs),
            'best_epoch': best_epoch_val,
            'last_epoch': last_epoch_val,
            'saved_epoch': int(saved_epoch_val),
            'auroc': best.get('auroc', float('nan')),
            'auprc': best.get('auprc', float('nan')),
            'accuracy': best.get('accuracy', float('nan')),
            'macro_f1': best.get('macro_f1', float('nan')),
            'last_auroc': dict(best.get('last_metrics', {}) or {}).get('auroc', float('nan')),
            'last_auprc': dict(best.get('last_metrics', {}) or {}).get('auprc', float('nan')),
            'last_accuracy': dict(best.get('last_metrics', {}) or {}).get('accuracy', float('nan')),
            'last_macro_f1': dict(best.get('last_metrics', {}) or {}).get('macro_f1', float('nan')),
            'saved_auroc': (dict(best.get('last_metrics', {}) or {}).get('auroc', float('nan')) if saved_checkpoint == 'last' else best.get('auroc', float('nan'))),
            'saved_auprc': (dict(best.get('last_metrics', {}) or {}).get('auprc', float('nan')) if saved_checkpoint == 'last' else best.get('auprc', float('nan'))),
            'saved_accuracy': (dict(best.get('last_metrics', {}) or {}).get('accuracy', float('nan')) if saved_checkpoint == 'last' else best.get('accuracy', float('nan'))),
            'saved_macro_f1': (dict(best.get('last_metrics', {}) or {}).get('macro_f1', float('nan')) if saved_checkpoint == 'last' else best.get('macro_f1', float('nan'))),
            'n_train': len(train_ids),
            'n_val': len(val_ids)
        })

        # 每个癌种完成后就更新一次顶层汇总（防止中途异常导致汇总缺失）
        _loco_write_summaries(base_loco_dir)

    # 结束后再刷新一次顶层汇总，并返回从磁盘收集到的结果（包含 resume 跳过的癌种）
    _loco_write_summaries(base_loco_dir)
    final_rows, _ = _loco_collect_completed(base_loco_dir)
    return {'loco_results': final_rows, 'base_dir': base_loco_dir}


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
