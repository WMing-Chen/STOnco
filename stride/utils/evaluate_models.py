#!/usr/bin/env python3
"""
汇总多个模型的批量推理结果，计算扩展评估指标并生成对比报告；
可选：基于生成的比较结果绘制可视化图表。

重构版本特性：
- 支持 --model NAME=CSV 灵活输入任意数量模型
- 计算10项分类指标：Accuracy, F1, Balanced_Accuracy, AUROC, AUPRC, Precision, Specificity, MCC, FPR, FNR
- 每个模型输出独立CSV文件包含所有指标  
- 默认可视化格式：png,pdf；Overall图为 Accuracy/Balanced_Accuracy/Precision 三子图
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

# =========================
# 可视化配置
# =========================
MODEL_COLORS = {
    'SAGE': '#ff7f0e',   # orange
    'GATv2': '#1f77b4',  # blue
    'GCN': '#2ca02c',    # green
}

# 动态颜色调色板（当遇到未知模型名时按需分配）
DEFAULT_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def ensure_model_colors(models: List[str]):
    """确保每个模型名都有颜色分配，对于未知模型使用默认调色板按需分配。"""
    used_colors = set(MODEL_COLORS.values())
    palette_iter = (c for c in DEFAULT_PALETTE if c not in used_colors)
    for m in models:
        if m not in MODEL_COLORS:
            try:
                MODEL_COLORS[m] = next(palette_iter)
            except StopIteration:
                # 若调色板耗尽则循环使用
                MODEL_COLORS[m] = DEFAULT_PALETTE[len(MODEL_COLORS) % len(DEFAULT_PALETTE)]
# 更新指标列表：Overall图显示的3个主要指标
OVERALL_METRICS = ['accuracy', 'balanced_accuracy', 'precision']
ALL_METRICS = ['accuracy', 'f1', 'balanced_accuracy', 'auroc', 'auprc', 
               'precision', 'specificity', 'mcc', 'fpr', 'fnr']


def detect_models(columns: List[str]) -> List[str]:
    """从DataFrame列名检测可用模型"""
    models = []
    for name in MODEL_COLORS.keys():
        if any(col.startswith(f'{name}_') for col in columns):
            models.append(name)
    # 保持顺序：SAGE, GATv2, GCN
    return [m for m in MODEL_COLORS.keys() if m in models]


def save_figure(fig, out_base: Path, fmts: List[str], dpi: int = 180):
    """保存图形到多种格式"""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in fmts:
        if fmt.lower() in ('png', 'jpg', 'jpeg'):
            fig.savefig(out_base.with_suffix('.' + fmt), dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(out_base.with_suffix('.' + fmt), bbox_inches='tight')


def sort_slides_by_metric(df_slides: pd.DataFrame, models: List[str], metric: str) -> pd.DataFrame:
    """按指标平均值对切片排序"""
    mean_vals = df_slides[[f'{m}_{metric}' for m in models]].mean(axis=1)
    return df_slides.assign(_mean_metric=mean_vals).sort_values('_mean_metric', ascending=True).drop(columns=['_mean_metric'])


def plot_overall_bars(df: pd.DataFrame, models: List[str], out_dir: Path, fmts: List[str], dpi: int):
    """绘制Overall总览柱状图：Accuracy/Balanced_Accuracy/Precision三子图"""
    import matplotlib.pyplot as plt
    overall = df[df['slide_id'] == 'OVERALL']
    if overall.empty:
        raise ValueError('OVERALL 行未找到，无法绘制 Overall 图')
    row = overall.iloc[0]

    # 调整长宽比：更长一些，稍微更高，便于阅读
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=False)
    for ax, metric in zip(axes, OVERALL_METRICS):
        vals = [row[f'{m}_{metric}'] for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        bars = ax.bar(models, vals, color=colors)
        ax.set_title(metric.replace('_', ' ').title())
        # Overall 图自适应上限：在最大值基础上增加 15%，便于顶部数值显示
        max_val = max(vals) if len(vals) > 0 else 1.0
        ax.set_ylim(0, max_val * 1.15)
        ax.set_ylabel('Score')
        # 动态偏移，避免文字贴边
        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + y_offset, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    n_spots = int(row['n_spots']) if 'n_spots' in overall.columns else None
    suptitle = 'Overall Performance'
    if n_spots is not None:
        suptitle += f' (n_spots={n_spots:,})'
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, out_dir / 'comparison_overall_bars', fmts, dpi)
    plt.close(fig)


def plot_per_slide_grouped_bars(df: pd.DataFrame, models: List[str], metric: str, out_dir: Path, fmts: List[str], dpi: int):
    """绘制Per-slide分组柱状图"""
    import matplotlib.pyplot as plt
    slides = df[df['slide_id'] != 'OVERALL'].copy()
    if slides.empty:
        return
    slides = sort_slides_by_metric(slides, models, metric)

    slide_ids = slides['slide_id'].tolist()
    x = np.arange(len(slide_ids))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(10, len(slide_ids) * 0.6), 4.5))
    for i, m in enumerate(models):
        vals = slides[f'{m}_{metric}'].to_numpy()
        ax.bar(x + (i - (len(models)-1)/2) * width, vals, width=width, label=m, color=MODEL_COLORS[m])

    ax.set_xticks(x)
    ax.set_xticklabels(slide_ids, rotation=45, ha='right')
    # 移除 ylim 限制以便看清楚差别  
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Per-slide Grouped Bars ({metric.replace("_", " ").title()})')
    ax.legend(ncol=len(models), frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    save_figure(fig, out_dir / f'per_slide_bars_{metric}', fmts, dpi)
    plt.close(fig)


def plot_per_slide_heatmap(df: pd.DataFrame, models: List[str], metric: str, out_dir: Path, fmts: List[str], dpi: int):
    """绘制Per-slide指标热力图"""
    import matplotlib.pyplot as plt
    slides = df[df['slide_id'] != 'OVERALL'].copy()
    if slides.empty:
        return
    slides = sort_slides_by_metric(slides, models, metric)

    mat = slides[[f'{m}_{metric}' for m in models]].to_numpy()
    slide_ids = slides['slide_id'].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(slide_ids) * 0.45), 4.2))
    # 移除 vmin/vmax 限制以便看清楚差别
    im = ax.imshow(mat, aspect='auto', cmap='YlGnBu')

    ax.set_yticks(np.arange(len(slide_ids)))
    ax.set_yticklabels(slide_ids)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_xlabel('Model')
    ax.set_title(f'Per-slide Heatmap ({metric.replace("_", " ").title()})')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center', fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())
    fig.tight_layout()

    save_figure(fig, out_dir / f'heatmap_{metric}', fmts, dpi)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, models: List[str], metric: str, out_dir: Path, fmts: List[str], dpi: int):
    """绘制指标分布箱线图/小提琴图"""
    import matplotlib.pyplot as plt
    slides = df[df['slide_id'] != 'OVERALL'].copy()
    if slides.empty:
        return

    data = [slides[f'{m}_{metric}'].dropna().to_numpy() for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Boxplot（使用 tick_labels 以避免未来版本警告）
    bp = axes[0].boxplot(data, patch_artist=True, tick_labels=models)
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.7)
    # 移除 ylim 限制以便看清楚差别
    axes[0].set_ylabel(metric.replace('_', ' ').title())
    axes[0].set_title(f'Per-slide Distribution (Boxplot) - {metric.replace("_", " ").title()}')
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    # Violin plot
    parts = axes[1].violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        m = models[i]
        pc.set_facecolor(MODEL_COLORS[m])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for k in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if k in parts:
            parts[k].set_edgecolor('black')
            parts[k].set_linewidth(1.0)
    axes[1].set_xticks(np.arange(1, len(models) + 1))
    axes[1].set_xticklabels(models)
    # 移除 ylim 限制以便看清楚差别
    axes[1].set_ylabel(metric.replace('_', ' ').title())
    axes[1].set_title(f'Per-slide Distribution (Violin) - {metric.replace("_", " ").title()}')
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    save_figure(fig, out_dir / f'distribution_{metric}', fmts, dpi)
    plt.close(fig)


# =========================
# 指标计算逻辑
# =========================

def compute_extended_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """计算扩展的10项分类指标"""
    if len(y_true) == 0:
        return dict.fromkeys(ALL_METRICS, 0.0)
    
    y_pred = (y_prob >= threshold).astype(int)
    
    # 基础指标
    acc = accuracy_score(y_true, y_pred)
    
    # 处理混淆矩阵
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true))
    except:
        tp = fp = tn = fn = 0
    
    # 防止除零
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    
    # 衍生指标
    balanced_acc = (recall + specificity) / 2
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = 1 - specificity
    fnr = 1 - recall
    
    # MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        if np.isnan(mcc):
            mcc = 0.0
    except:
        mcc = 0.0
    
    # 阈值无关指标
    if len(np.unique(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
        except:
            auroc = auprc = 0.0
    else:
        auroc = auprc = 0.0
    
    return {
        'accuracy': acc,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'auroc': auroc,
        'auprc': auprc,
        'precision': precision,
        'specificity': specificity,
        'mcc': mcc,
        'fpr': fpr,
        'fnr': fnr
    }


def parse_model_arguments(args) -> Dict[str, str]:
    """解析模型输入参数，支持新格式和兼容旧格式"""
    models = {}
    
    # 新格式：--model NAME=CSV
    if hasattr(args, 'model') and args.model:
        legacy_used = any([args.sage_csv, args.gatv2_csv, args.gcn_csv])
        if legacy_used:
            print("Warning: 检测到同时使用 --model 和旧格式参数，优先使用 --model")
        
        for model_spec in args.model:
            if '=' not in model_spec:
                raise ValueError(f"--model 参数格式错误：{model_spec}，应为 NAME=CSV")
            name, csv_path = model_spec.split('=', 1)
            name = name.strip()
            models[name] = csv_path
    else:
        # 兼容旧格式
        if args.sage_csv:
            models['SAGE'] = args.sage_csv
        if args.gatv2_csv:
            models['GATv2'] = args.gatv2_csv
        if args.gcn_csv:
            models['GCN'] = args.gcn_csv
    
    if not models:
        raise ValueError("至少需要指定一个模型，使用 --model NAME=CSV 或旧格式参数")
    
    return models


def main():
    parser = argparse.ArgumentParser(description='评估多个模型的预测结果并可选绘图')
    
    # 新模型输入方式
    parser.add_argument('--model', action='append', help='模型规格：NAME=CSV，可重复多次。例：--model GCN=path/to/gcn.csv')
    parser.add_argument('--out_dir', default='.', help='输出目录（单模型CSV和图表的保存位置）')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    
    # 兼容旧参数
    parser.add_argument('--sage_csv', default='', help='[兼容] SAGE模型预测结果CSV')
    parser.add_argument('--gatv2_csv', default='', help='[兼容] GATv2模型预测结果CSV')
    parser.add_argument('--gcn_csv', default='', help='[兼容] GCN模型预测结果CSV')
    parser.add_argument('--out_csv', default='', help='[兼容] 输出比较结果CSV（已弃用，改为单模型CSV）')
    
    # 绘图控制参数
    parser.add_argument('--plot', action='store_true', help='是否根据结果绘制图表')
    parser.add_argument('--plot_out_dir', default=None, help='图表输出目录（默认与 out_dir 相同）')
    parser.add_argument('--plot_metric', default='accuracy', choices=ALL_METRICS, 
                       help='Per-slide 图表使用的指标')
    parser.add_argument('--plot_formats', default='svg,png', 
                       help='图表导出格式，逗号分隔，例如 svg,png')
    parser.add_argument('--plot_dpi', type=int, default=180, help='位图格式导出DPI')
    
    args = parser.parse_args()
    
    # 解析模型输入
    models_config = parse_model_arguments(args)
    
    # 读取预测结果
    models_pred = {}
    for model_name, csv_path in models_config.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"模型 {model_name} 的预测文件不存在: {csv_path}")
        models_pred[model_name] = pd.read_csv(csv_path)
        print(f"已加载模型 {model_name}: {csv_path}")
    
    # 获取所有切片ID
    all_slide_ids = set()
    for pred_df in models_pred.values():
        all_slide_ids.update(pred_df['sample_id'].unique())
    slide_ids = sorted(all_slide_ids)
    
    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个模型生成独立的指标CSV
    models_results = {}
    
    for model_name, pred_df in models_pred.items():
        print(f"\n处理模型: {model_name}")
        
        # 计算每个切片的指标
        slide_results = []
        overall_y_true = []
        overall_y_prob = []
        
        for slide_id in slide_ids:
            slide_pred = pred_df[pred_df['sample_id'] == slide_id].copy()
            
            if len(slide_pred) == 0:
                print(f"  Warning: 切片 {slide_id} 无预测数据")
                slide_metrics = {'slide_id': slide_id, 'n_spots': 0}
                slide_metrics.update(dict.fromkeys(ALL_METRICS, 0.0))
                slide_results.append(slide_metrics)
                continue
            
            # 过滤有效标签
            valid_pred = slide_pred[slide_pred['y_true'] >= 0]
            if len(valid_pred) == 0:
                print(f"  Warning: 切片 {slide_id} 无有效标签")
                slide_metrics = {'slide_id': slide_id, 'n_spots': 0}
                slide_metrics.update(dict.fromkeys(ALL_METRICS, 0.0))
                slide_results.append(slide_metrics)
                continue
            
            # 计算切片指标
            y_true = valid_pred['y_true'].values
            y_prob = valid_pred['p_tumor'].values
            
            metrics = compute_extended_metrics(y_true, y_prob, args.threshold)
            slide_metrics = {
                'slide_id': slide_id,
                'n_spots': len(valid_pred),
                **metrics
            }
            slide_results.append(slide_metrics)
            
            # 累积用于Overall计算
            overall_y_true.extend(y_true)
            overall_y_prob.extend(y_prob)
        
        # 计算Overall指标
        overall_metrics = {'slide_id': 'OVERALL', 'n_spots': len(overall_y_true)}
        if len(overall_y_true) > 0:
            overall_metrics.update(compute_extended_metrics(
                np.array(overall_y_true), 
                np.array(overall_y_prob), 
                args.threshold
            ))
        else:
            overall_metrics.update(dict.fromkeys(ALL_METRICS, 0.0))
        
        # 组装结果DataFrame
        all_results = [overall_metrics] + slide_results
        model_df = pd.DataFrame(all_results)
        
        # 保存单模型CSV
        model_csv_path = out_dir / f'{model_name.lower()}_metrics.csv'
        model_df.to_csv(model_csv_path, index=False, float_format='%.4f')
        print(f"  已保存: {model_csv_path}")
        
        # 存储用于绘图
        models_results[model_name] = model_df
    
    print(f"\n所有单模型CSV已保存至: {out_dir}")
    
    # 可选绘图
    if args.plot:
        plot_out_dir = Path(args.plot_out_dir) if args.plot_out_dir else out_dir
        fmts = [f.strip() for f in args.plot_formats.split(',') if f.strip()]
        metric = args.plot_metric
        
        # 构建用于绘图的综合DataFrame（仅用于绘图，不保存）
        combined_results = []
        model_names = list(models_results.keys())
        
        # 获取所有slide_id（包括OVERALL）
        all_slide_set = set()
        for model_df in models_results.values():
            all_slide_set.update(model_df['slide_id'])
        all_slides = ['OVERALL'] + sorted([s for s in all_slide_set if s != 'OVERALL'])
        
        for slide_id in all_slides:
            row = {'slide_id': slide_id, 'n_spots': 0}
            for model_name in model_names:
                model_df = models_results[model_name]
                slide_row = model_df[model_df['slide_id'] == slide_id]
                
                if not slide_row.empty:
                    slide_data = slide_row.iloc[0]
                    if slide_id == 'OVERALL':
                        row['n_spots'] = slide_data.get('n_spots', 0)
                    for m in ALL_METRICS:
                        row[f'{model_name}_{m}'] = slide_data.get(m, 0.0)
                else:
                    # 缺失数据填0
                    for m in ALL_METRICS:
                        row[f'{model_name}_{m}'] = 0.0
            
            combined_results.append(row)
        
        combined_df = pd.DataFrame(combined_results)
        
        # 直接使用传入的模型名顺序，并为所有模型确保颜色可用
        models_available = model_names
        ensure_model_colors(models_available)
        
        # 检查绘图所需列
        missing = [f'{m}_{metric}' for m in models_available if f'{m}_{metric}' not in combined_df.columns]
        if missing:
            raise ValueError('用于绘图的列缺失: ' + ', '.join(missing))
        
        print(f"\n开始绘制图表到: {plot_out_dir}")
        
        # 1) Overall 总览柱状图（3子图：accuracy/balanced_accuracy/precision）
        plot_overall_bars(combined_df, models_available, plot_out_dir, fmts, args.plot_dpi)
        print("  - comparison_overall_bars 已生成")
        
        # 2) Per-slide 分组柱状图（选定指标）
        plot_per_slide_grouped_bars(combined_df, models_available, metric, plot_out_dir, fmts, args.plot_dpi)
        print(f"  - per_slide_bars_{metric} 已生成")
        
        # 3) Per-slide 指标热力图（选定指标）
        plot_per_slide_heatmap(combined_df, models_available, metric, plot_out_dir, fmts, args.plot_dpi)
        print(f"  - heatmap_{metric} 已生成")
        
        # 4) 指标分布箱线图/小提琴图（选定指标）
        plot_distribution(combined_df, models_available, metric, plot_out_dir, fmts, args.plot_dpi)
        print(f"  - distribution_{metric} 已生成")
        
        print(f"\n可视化图表已保存至: {plot_out_dir}")


if __name__ == '__main__':
    main()