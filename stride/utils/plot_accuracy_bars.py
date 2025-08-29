#!/usr/bin/env python3
"""
为每个模型绘制柱状图，展示各验证切片的准确性
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def load_true_labels_from_valdir(val_root: str, slide_id: str, x_col='row', y_col='col', label_col='true_label'):
    """从验证集目录读取切片的真实标签"""
    slide_dir = os.path.join(val_root, slide_id)
    coord_csv = None
    if os.path.isdir(slide_dir):
        for f in os.listdir(slide_dir):
            if f.lower().endswith('coordinates.csv'):
                coord_csv = os.path.join(slide_dir, f)
                break
    if coord_csv is None:
        cand = os.path.join(slide_dir, f"{slide_id}_coordinates.csv")
        if os.path.exists(cand):
            coord_csv = cand
    if coord_csv is None:
        raise FileNotFoundError(f"Coordinates CSV not found for slide '{slide_id}' under {slide_dir}")

    df = pd.read_csv(coord_csv)
    cmap = {c.lower(): c for c in df.columns}
    
    def _resolve(name: str) -> str:
        c = cmap.get(name.lower())
        if c is None:
            raise ValueError(f"Column '{name}' not found in {coord_csv}. Available: {list(df.columns)}")
        return c
    
    x_name = _resolve(x_col)
    y_name = _resolve(y_col)
    lbl_name = _resolve(label_col)
    
    xr = pd.to_numeric(df[x_name], errors='coerce').round().astype('Int64')
    yr = pd.to_numeric(df[y_name], errors='coerce').round().astype('Int64')
    yl = pd.to_numeric(df[lbl_name], errors='coerce').fillna(0).astype(int)
    
    return xr, yr, yl


def compute_slide_accuracy(pred_csv: str, slide_id: str, threshold=0.5):
    """计算指定切片的准确率"""
    
    # 读取预测结果
    pred_df = pd.read_csv(pred_csv)
    slide_preds = pred_df[pred_df['sample_id'] == slide_id].copy()
    
    if len(slide_preds) == 0:
        print(f"Warning: No predictions found for slide {slide_id}")
        return 0.0, 0
    
    # 过滤有效标签
    valid_mask = slide_preds['y_true'] >= 0
    valid_data = slide_preds[valid_mask]
    
    if len(valid_data) == 0:
        print(f"Warning: No valid labels found for slide {slide_id}")
        return 0.0, 0
    
    # 计算准确率
    y_true = valid_data['y_true'].values
    y_pred = (valid_data['p_tumor'].values >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    return accuracy, len(valid_data)


def plot_accuracy_bars(model_name: str, pred_csv: str, slide_ids: list, out_path: str, threshold=0.5):
    """为指定模型绘制各切片准确率柱状图（跳过 n=0 的切片，均值不含这些切片）"""
    
    accuracies = []
    n_spots_list = []
    kept_slide_ids = []
    
    for slide_id in slide_ids:
        acc, n_spots = compute_slide_accuracy(pred_csv, slide_id, threshold=threshold)
        if n_spots == 0:
            print(f"Skip slide {slide_id} due to missing/invalid labels (n=0).")
            continue
        accuracies.append(acc)
        n_spots_list.append(n_spots)
        kept_slide_ids.append(slide_id)
    
    # 若没有任何有效切片，生成空图以避免报错
    if len(kept_slide_ids) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No valid slides to plot', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved empty accuracy bar plot (no valid slides) to: {out_path}")
        return 0.0, 0
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 使用tab20颜色映射为每个样本分配不同颜色
    cmap = plt.cm.tab20
    colors = [cmap(i / len(kept_slide_ids)) for i in range(len(kept_slide_ids))]
    
    bars = ax.bar(range(len(kept_slide_ids)), accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 在柱子上标注准确率百分比
    for i, (acc, n_spots) in enumerate(zip(accuracies, n_spots_list)):
        ax.text(i, acc + 0.01, f'{acc*100:.1f}%\n(n={n_spots})', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 设置图表属性
    ax.set_xlabel('Validation Slides', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} Model Accuracy per Validation Slide', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(kept_slide_ids)))
    ax.set_xticklabels(kept_slide_ids, rotation=45, ha='right')
    ax.set_ylim(0, 1.2)  # 增加10%的上限空间
    ax.grid(axis='y', alpha=0.3)
    
    # 添加水平参考线
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Level (50%)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Level (80%)')
    ax.legend(loc='upper right')
    
    # 添加整体统计信息（均值仅对有效切片计算）
    mean_acc = np.mean(accuracies)
    total_spots = sum(n_spots_list)
    ax.text(0.02, 0.98, f'Mean Accuracy: {mean_acc*100:.1f}%\nTotal Spots: {total_spots}', 
            transform=ax.transAxes, va='top', ha='left', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved accuracy bar plot for {model_name} to: {out_path}")
    return mean_acc, total_spots


def main():
    parser = argparse.ArgumentParser(description='为每个模型绘制验证切片准确率柱状图')
    parser.add_argument('--pred_csv', required=True, help='模型预测结果CSV文件')
    parser.add_argument('--model_name', required=True, help='模型名称 (用于标题)')
    parser.add_argument('--out_path', required=True, help='输出图片路径 (PNG格式)')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    
    args = parser.parse_args()
    
    # 从预测CSV中获取slide_ids
    pred_df = pd.read_csv(args.pred_csv)
    slide_ids = sorted(set(pred_df['sample_id']))
    
    print(f"Found {len(slide_ids)} slides: {slide_ids}")
    
    # 绘制柱状图
    mean_acc, total_spots = plot_accuracy_bars(
        model_name=args.model_name,
        pred_csv=args.pred_csv,
        slide_ids=slide_ids,
        out_path=args.out_path,
        threshold=args.threshold
    )
    
    print(f"\nSummary for {args.model_name}:")
    print(f"  Mean accuracy: {mean_acc*100:.2f}%")
    print(f"  Total spots: {total_spots}")


if __name__ == '__main__':
    main()