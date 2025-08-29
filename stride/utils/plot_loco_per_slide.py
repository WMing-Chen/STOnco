#!/usr/bin/env python3
"""
从 LOCO 每切片指标汇总表（loco_per_slide.csv）按癌种进行可视化：
- 箱线图 + 蜂群图（默认绘制 Accuracy，可选其他: auroc, auprc, accuracy, macro_f1, pos_rate）
- 指标-癌种 热力图（默认四项：auroc, auprc, accuracy, macro_f1；聚合方式默认 median）
- AUROC vs pos_rate 散点图（每个点一张切片；颜色=癌种；点大小 ~ n_nodes）

用法示例：
python plot_loco_per_slide.py \
  --loco_csv test_train0822_SC3331genes_68sample/loco_eval/loco_per_slide.csv \
  --out_dir test_train0822_SC3331genes_68sample/loco_eval/visualizations \
  --box_metric accuracy \
  --heatmap_metrics auroc,auprc,accuracy,macro_f1

说明：
- 会自动创建输出目录；默认导出格式: svg,png；DPI=180（仅位图格式生效）。
- 会自动忽略缺失值；若某癌种没有有效值，将在对应图中跳过或显示空。
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 支持的指标列（大小写不敏感）
KNOWN_COLS = [
    'cancer_type', 'slide_id', 'n_nodes', 'n_pos', 'n_neg', 'pos_rate',
    'auroc', 'auprc', 'accuracy', 'macro_f1', 'best_epoch'
]


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    cmap = {c.lower(): c for c in df.columns}
    resolved = {}
    for k in KNOWN_COLS:
        if k.lower() in cmap:
            resolved[k] = cmap[k.lower()]
    required = ['cancer_type', 'slide_id']
    for r in required:
        if r not in resolved:
            raise ValueError(f"缺少必要列: {r}。CSV中包含: {list(df.columns)}")
    return resolved


def save_figure(fig, out_base: Path, fmts: List[str], dpi: int = 180):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in fmts:
        if fmt.lower() in ('png', 'jpg', 'jpeg'):
            fig.savefig(out_base.with_suffix('.' + fmt), dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(out_base.with_suffix('.' + fmt), bbox_inches='tight')


def color_map_for_categories(categories: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    uniq = sorted(set(categories))
    cmap = plt.cm.tab20
    colors = {cat: cmap(i / max(1, len(uniq))) for i, cat in enumerate(uniq)}
    return colors


def plot_box_swarm(df: pd.DataFrame, cols: Dict[str, str], metric: str, out_dir: Path, fmts: List[str], dpi: int):
    if metric not in cols and metric not in df.columns:
        # 如果列未在已解析中，但原df中存在同名（大小写不敏感）也允许
        lower_map = {c.lower(): c for c in df.columns}
        if metric.lower() in lower_map:
            metric_col = lower_map[metric.lower()]
        else:
            raise ValueError(f"箱线图指标列不存在: {metric}")
    else:
        metric_col = cols.get(metric, metric)

    cancer_col = cols['cancer_type']

    # 准备数据
    g = df[[cancer_col, metric_col]].copy()
    g = g.dropna(subset=[metric_col])
    if g.empty:
        print(f"[Box+Swarm] 指标 {metric} 无有效数据，跳过绘制。")
        return

    groups = []
    labels = []
    for cancer, sub in g.groupby(cancer_col):
        vals = sub[metric_col].astype(float).values
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(cancer)

    if len(groups) == 0:
        print(f"[Box+Swarm] 无有效癌种组，跳过绘制。")
        return

    colors = color_map_for_categories(labels)

    # 绘图
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))

    # 箱线图
    bp = ax.boxplot(groups, patch_artist=True, positions=np.arange(1, len(groups) + 1))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[labels[i]])
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)

    # 蜂群点（使用抖动）
    for i, vals in enumerate(groups, start=1):
        x = np.random.uniform(low=i - 0.18, high=i + 0.18, size=len(vals))
        ax.scatter(x, vals, s=22, color=colors[labels[i-1]], edgecolors='black', linewidths=0.3, alpha=0.85)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f"LOCO Per-cancer Distribution - {metric.replace('_', ' ').title()}")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    save_figure(fig, out_dir / f"box_swarm_{metric}", fmts, dpi)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, cols: Dict[str, str], metrics: List[str], agg: str, out_dir: Path, fmts: List[str], dpi: int):
    cancer_col = cols['cancer_type']
    metrics_cols = []
    lower_map = {c.lower(): c for c in df.columns}
    for m in metrics:
        if m in cols:
            metrics_cols.append((m, cols[m]))
        elif m.lower() in lower_map:
            metrics_cols.append((m, lower_map[m.lower()]))
        else:
            print(f"[Heatmap] 指标列不存在，跳过: {m}")
    if not metrics_cols:
        print("[Heatmap] 无可用指标列，跳过绘制。")
        return

    if agg not in ('mean', 'median'):
        agg = 'median'

    # 聚合
    if agg == 'mean':
        agg_df = df.groupby(cancer_col)[[c for _, c in metrics_cols]].mean(numeric_only=True)
    else:
        agg_df = df.groupby(cancer_col)[[c for _, c in metrics_cols]].median(numeric_only=True)

    if agg_df.empty:
        print("[Heatmap] 聚合结果为空，跳过绘制。")
        return

    agg_df = agg_df.rename(columns={c: m for m, c in metrics_cols})
    cancers = agg_df.index.tolist()
    mat = agg_df.to_numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(metrics_cols) * 1.2), max(4, len(cancers) * 0.42)))
    im = ax.imshow(mat, aspect='auto', cmap='YlGnBu')

    ax.set_xticks(np.arange(len(metrics_cols)))
    ax.set_xticklabels([m for m, _ in metrics_cols])
    ax.set_yticks(np.arange(len(cancers)))
    ax.set_yticklabels(cancers)
    ax.set_xlabel('Metric')
    ax.set_title(f"Per-cancer Heatmap ({agg.title()})")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha='center', va='center', fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Score')

    fig.tight_layout()
    save_figure(fig, out_dir / f"heatmap_{agg}", fmts, dpi)
    plt.close(fig)


def plot_scatter_auroc_posrate(df: pd.DataFrame, cols: Dict[str, str], out_dir: Path, fmts: List[str], dpi: int,
                                size_by_nodes: bool = True, size_min: int = 18, size_max: int = 220):
    cancer_col = cols['cancer_type']
    posrate_col = cols.get('pos_rate', None)
    auroc_col = cols.get('auroc', None)
    n_nodes_col = cols.get('n_nodes', None)

    if posrate_col is None or auroc_col is None:
        print("[Scatter] 需要列 pos_rate 与 auroc，缺失则跳过绘制。")
        return

    data = df[[cancer_col, posrate_col, auroc_col] + ([n_nodes_col] if n_nodes_col else [])].copy()
    data = data.dropna(subset=[posrate_col, auroc_col])
    if data.empty:
        print("[Scatter] 无有效 AUROC/pos_rate 数据，跳过绘制。")
        return

    # 颜色
    colors = color_map_for_categories(sorted(data[cancer_col].unique()))

    # 点大小
    if size_by_nodes and (n_nodes_col is not None) and (n_nodes_col in data.columns):
        nn = data[n_nodes_col].astype(float).values
        if len(nn) > 0:
            nn = np.maximum(nn, 1.0)
            s = np.interp(np.sqrt(nn), (np.sqrt(nn).min(), np.sqrt(nn).max()), (size_min, size_max))
        else:
            s = np.full(len(data), 60)
    else:
        s = np.full(len(data), 60)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for cancer, sub in data.groupby(cancer_col):
        ax.scatter(sub[posrate_col], sub[auroc_col], s=s if isinstance(s, (int, float)) else s[sub.index],
                   color=colors[cancer], edgecolors='black', linewidths=0.4, alpha=0.85, label=cancer)

    ax.set_xlabel('pos_rate')
    ax.set_ylabel('AUROC')
    ax.set_title('AUROC vs pos_rate (per slide)')
    ax.grid(True, linestyle='--', alpha=0.35)

    # 参考线
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    ax.legend(title='Cancer', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    save_figure(fig, out_dir / "scatter_auroc_vs_pos_rate", fmts, dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='LOCO每切片指标表的分癌种可视化（箱线+蜂群、热力图、散点）')
    parser.add_argument('--loco_csv', required=True, help='loco_per_slide.csv 路径')
    parser.add_argument('--out_dir', required=False, default=None, help='输出目录（默认：CSV同级目录下 visualizations 子目录）')
    parser.add_argument('--box_metric', default='accuracy', help='箱线+蜂群图的指标（默认 accuracy）')
    parser.add_argument('--heatmap_metrics', default='auroc,auprc,accuracy,macro_f1', help='热力图指标列表，逗号分隔')
    parser.add_argument('--heatmap_agg', default='median', choices=['mean', 'median'], help='热力图聚合方式')
    parser.add_argument('--plot_formats', default='svg,png', help='导出格式，逗号分隔，如 svg,png,pdf')
    parser.add_argument('--plot_dpi', type=int, default=180, help='位图DPI（对png/jpg生效）')
    parser.add_argument('--no_box_swarm', action='store_true', help='不绘制箱线+蜂群图')
    parser.add_argument('--no_heatmap', action='store_true', help='不绘制热力图')
    parser.add_argument('--no_scatter', action='store_true', help='不绘制AUROC vs pos_rate散点图')

    args = parser.parse_args()

    loco_csv = Path(args.loco_csv)
    if not loco_csv.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {loco_csv}")

    out_dir = Path(args.out_dir) if args.out_dir else loco_csv.parent / 'visualizations'
    fmts = [s.strip() for s in args.plot_formats.split(',') if s.strip()]

    # 读取
    df = pd.read_csv(loco_csv)
    cols = resolve_columns(df)

    # 绘制
    if not args.no_box_swarm:
        try:
            plot_box_swarm(df, cols, args.box_metric, out_dir, fmts, args.plot_dpi)
        except Exception as e:
            print(f"绘制箱线+蜂群图失败: {e}")

    if not args.no_heatmap:
        metrics = [m.strip() for m in args.heatmap_metrics.split(',') if m.strip()]
        try:
            plot_heatmap(df, cols, metrics, args.heatmap_agg, out_dir, fmts, args.plot_dpi)
        except Exception as e:
            print(f"绘制热力图失败: {e}")

    if not args.no_scatter:
        try:
            plot_scatter_auroc_posrate(df, cols, out_dir, fmts, args.plot_dpi)
        except Exception as e:
            print(f"绘制散点图失败: {e}")

    print(f"可视化完成，输出目录：{out_dir}")


if __name__ == '__main__':
    main()