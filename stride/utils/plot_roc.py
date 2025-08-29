#!/usr/bin/env python3
"""
从预测结果CSV绘制ROC曲线
- 支持 Overall（合并所有切片）ROC
- 支持 Per-slide（每张切片单独）ROC

输入CSV需包含以下列（不区分大小写）：
- sample_id  切片/样本ID（Per-slide模式需要）
- y_true     真实标签（0/1；忽略 <0 的无效标签）
- p_tumor    属于正类(1)的预测概率

用法示例：
# 仅绘制 overall ROC（默认保存为 svg 和 png）
python plot_roc.py --pred_csv path/to/val_preds.csv --mode overall --out_path out/roc_overall

# 绘制每张切片的ROC到指定目录
python plot_roc.py --pred_csv path/to/val_preds.csv --mode per-slide --out_dir out/roc_per_slide

# 同时绘制 overall 与 per-slide
python plot_roc.py --pred_csv path/to/val_preds.csv --mode both --out_path out/roc_overall --out_dir out/roc_per_slide
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """在DataFrame列中解析出 sample_id, y_true, p_tumor 的真实列名（大小写不敏感）。"""
    cmap = {c.lower(): c for c in df.columns}
    def pick(*candidates: str) -> str:
        for name in candidates:
            if name.lower() in cmap:
                return cmap[name.lower()]
        raise ValueError(f"CSV缺少列：{candidates}")
    sample_col = pick('sample_id')
    ytrue_col = pick('y_true')
    prob_col = pick('p_tumor')
    return sample_col, ytrue_col, prob_col


def _ensure_out_base(out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)


def _save_figure(fig, out_base: Path, fmts, dpi: int):
    _ensure_out_base(out_base)
    for fmt in fmts:
        fmt = fmt.lower()
        if fmt in ('png', 'jpg', 'jpeg'):
            fig.savefig(out_base.with_suffix('.' + fmt), dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(out_base.with_suffix('.' + fmt), bbox_inches='tight')


def plot_overall_roc(pred_df: pd.DataFrame, out_base: Path, formats=('svg','png'), dpi: int = 180, title: str | None = None):
    """合并全部有效样本绘制一张 ROC 曲线。"""
    sample_col, ytrue_col, prob_col = _resolve_columns(pred_df)
    df = pred_df.copy()
    # 过滤无效标签
    df = df[pd.to_numeric(df[ytrue_col], errors='coerce') >= 0]
    if df.empty:
        raise ValueError('没有可用的有效标签(y_true>=0)用于绘制ROC。')

    y_true = df[ytrue_col].astype(int).to_numpy()
    y_prob = pd.to_numeric(df[prob_col], errors='coerce').fillna(0.0).to_numpy()

    # 至少需要同时包含两类
    if len(np.unique(y_true)) < 2:
        raise ValueError('整体数据只有单一类别，无法计算ROC。')

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, ls='--', label='Chance')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title if title else 'Overall ROC')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.3, ls='--')
    fig.tight_layout()

    _save_figure(fig, out_base, formats, dpi)
    plt.close(fig)
    print(f"Saved overall ROC to base: {out_base}")


def plot_per_slide_roc(pred_df: pd.DataFrame, out_dir: Path, formats=('svg','png'), dpi: int = 180):
    """为每个切片绘制单独的ROC曲线。遇到只有单一类别的切片会跳过。"""
    sample_col, ytrue_col, prob_col = _resolve_columns(pred_df)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_slides = 0
    n_drawn = 0
    skipped = []

    for slide_id, g in pred_df.groupby(sample_col):
        n_slides += 1
        df = g.copy()
        df = df[pd.to_numeric(df[ytrue_col], errors='coerce') >= 0]
        if df.empty:
            skipped.append((slide_id, 'no_valid_labels'))
            continue
        y_true = df[ytrue_col].astype(int).to_numpy()
        y_prob = pd.to_numeric(df[prob_col], errors='coerce').fillna(0.0).to_numpy()
        if len(np.unique(y_true)) < 2:
            skipped.append((slide_id, 'single_class'))
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.plot(fpr, tpr, color='#2ca02c', lw=2, label=f'AUC={auc_val:.3f}')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, ls='--')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC - {slide_id}')
        ax.legend(loc='lower right', frameon=False)
        ax.grid(alpha=0.3, ls='--')
        fig.tight_layout()

        out_base = out_dir / f"roc_{slide_id}"
        _save_figure(fig, out_base, formats, dpi)
        plt.close(fig)
        n_drawn += 1

    print(f"Per-slide ROC saved to: {out_dir} (drawn={n_drawn}, total_slides={n_slides}, skipped={len(skipped)})")
    if skipped:
        reasons = {}
        for sid, why in skipped:
            reasons.setdefault(why, 0)
            reasons[why] += 1
        print("Skipped details:")
        for k, v in reasons.items():
            print(f"  - {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description='从预测CSV绘制ROC曲线（overall 或 per-slide）')
    parser.add_argument('--pred_csv', required=True, help='预测结果CSV（需包含 sample_id, y_true, p_tumor）')
    parser.add_argument('--mode', choices=['overall', 'per-slide', 'both'], default='overall', help='绘制模式')
    parser.add_argument('--out_path', default='roc_overall', help='overall 模式的输出基名（不含扩展名）')
    parser.add_argument('--out_dir', default='roc_per_slide', help='per-slide 模式的输出目录')
    parser.add_argument('--plot_formats', default='svg,png', help='导出格式，逗号分隔，例如 svg,png')
    parser.add_argument('--plot_dpi', type=int, default=180, help='位图格式导出DPI')
    parser.add_argument('--title', default='', help='overall 图标题（可选）')

    args = parser.parse_args()

    if not os.path.exists(args.pred_csv):
        raise FileNotFoundError(f"预测CSV不存在: {args.pred_csv}")

    pred_df = pd.read_csv(args.pred_csv)
    fmts = [f.strip() for f in args.plot_formats.split(',') if f.strip()]

    if args.mode in ('overall', 'both'):
        title = args.title if args.title else None
        plot_overall_roc(pred_df, Path(args.out_path), formats=fmts, dpi=args.plot_dpi, title=title)

    if args.mode in ('per-slide', 'both'):
        plot_per_slide_roc(pred_df, Path(args.out_dir), formats=fmts, dpi=args.plot_dpi)


if __name__ == '__main__':
    main()