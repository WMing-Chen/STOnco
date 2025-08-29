#!/usr/bin/env python3
"""
生成 10x Visium 空间转录组模拟数据
用于测试 Spotonco GNN 分类模型

生成规格：
- 45个切片：40个用于训练，5个用于推理
- 每个切片2000-3000个spot
- 肿瘤spot占比40%-70%
- 2000个基因：100个关键基因 + 300个辅助基因 + 1600个噪声基因
- 空间坐标模拟真实 Visium 布局
"""

import numpy as np
import os
import argparse
from typing import Tuple, List, Dict
import random

def set_random_seed(seed: int = 42):
    """设置随机种子确保结果可重现"""
    np.random.seed(seed)
    random.seed(seed)

def generate_visium_coordinates(n_spots: int, grid_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    生成类似 Visium 的六边形网格坐标
    
    Args:
        n_spots: spot数量
        grid_size: 网格大小 (width, height)
    
    Returns:
        coords: (n_spots, 2) 坐标数组
    """
    coords = []
    max_x, max_y = grid_size
    
    # 生成六边形网格模式
    row = 0
    while len(coords) < n_spots and row < max_y:
        # 奇数行偏移
        x_offset = 0.5 if row % 2 == 1 else 0.0
        
        col = 0
        while len(coords) < n_spots and col < max_x:
            x = col + x_offset
            y = row * 0.866  # 六边形网格的 y 间距
            
            # 添加小量随机噪声模拟真实位置变化
            x += np.random.normal(0, 0.1)
            y += np.random.normal(0, 0.1)
            
            coords.append([x, y])
            col += 1
        row += 1
    
    # 如果还不够，随机填充
    while len(coords) < n_spots:
        x = np.random.uniform(0, max_x)
        y = np.random.uniform(0, max_y * 0.866)
        coords.append([x, y])
    
    coords = np.array(coords[:n_spots])
    
    # 标准化到 [0, 1] 范围
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    
    return coords

def generate_tumor_labels(coords: np.ndarray, tumor_ratio: float) -> np.ndarray:
    """
    基于空间位置生成肿瘤标签，模拟肿瘤区域的空间聚集性
    
    Args:
        coords: spot坐标 (n_spots, 2)
        tumor_ratio: 肿瘤spot比例
    
    Returns:
        labels: (n_spots,) 二元标签，1=肿瘤，0=正常
    """
    n_spots = coords.shape[0]
    labels = np.zeros(n_spots, dtype=int)
    
    # 随机选择2-4个肿瘤中心
    n_centers = np.random.randint(2, 5)
    centers = coords[np.random.choice(n_spots, n_centers, replace=False)]
    
    # 为每个spot计算到最近肿瘤中心的距离
    distances = np.inf * np.ones(n_spots)
    for center in centers:
        dist_to_center = np.sqrt(np.sum((coords - center) ** 2, axis=1))
        distances = np.minimum(distances, dist_to_center)
    
    # 基于距离设置肿瘤概率（距离越近概率越高）
    max_dist = np.max(distances)
    tumor_probs = 1.0 - (distances / max_dist)  # 距离越近概率越高
    tumor_probs = tumor_probs ** 2  # 增强聚集效应
    
    # 调整概率使总体比例接近目标
    sorted_probs = np.sort(tumor_probs)[::-1]
    target_count = int(n_spots * tumor_ratio)
    threshold = sorted_probs[target_count - 1] if target_count <= n_spots else 0
    
    # 加入随机性
    final_probs = tumor_probs * 0.8 + np.random.random(n_spots) * 0.2
    labels = (final_probs >= threshold).astype(int)
    
    # 微调确保比例接近目标
    current_ratio = labels.mean()
    if abs(current_ratio - tumor_ratio) > 0.05:
        diff = int((tumor_ratio - current_ratio) * n_spots)
        if diff > 0:  # 需要更多肿瘤spot
            normal_indices = np.where(labels == 0)[0]
            if len(normal_indices) >= diff:
                flip_indices = np.random.choice(normal_indices, diff, replace=False)
                labels[flip_indices] = 1
        else:  # 需要更少肿瘤spot
            tumor_indices = np.where(labels == 1)[0]
            if len(tumor_indices) >= abs(diff):
                flip_indices = np.random.choice(tumor_indices, abs(diff), replace=False)
                labels[flip_indices] = 0
    
    return labels

def generate_gene_expression(n_spots: int, n_genes: int, labels: np.ndarray, 
                           key_genes: int = 100, helper_genes: int = 300) -> np.ndarray:
    """
    生成基因表达矩阵，模拟不同基因对肿瘤预测的重要性
    
    Args:
        n_spots: spot数量
        n_genes: 基因总数
        labels: 肿瘤标签
        key_genes: 关键基因数量（强预测能力）
        helper_genes: 辅助基因数量（中等预测能力）
    
    Returns:
        expression: (n_spots, n_genes) 基因表达矩阵
    """
    expression = np.zeros((n_spots, n_genes))
    
    # 基础表达水平（所有基因）
    base_expression = np.random.lognormal(mean=2.0, sigma=1.5, size=(n_spots, n_genes))
    expression = base_expression
    
    # 关键基因（0-99）：肿瘤和正常组织差异显著
    for i in range(key_genes):
        if np.random.random() > 0.5:
            # 肿瘤上调基因
            tumor_multiplier = np.random.uniform(2.0, 5.0)
            expression[labels == 1, i] *= tumor_multiplier
            # 添加一些噪声
            expression[:, i] *= np.random.lognormal(0, 0.3, n_spots)
        else:
            # 肿瘤下调基因
            tumor_multiplier = np.random.uniform(0.2, 0.6)
            expression[labels == 1, i] *= tumor_multiplier
            # 添加一些噪声
            expression[:, i] *= np.random.lognormal(0, 0.3, n_spots)
    
    # 辅助基因（100-399）：中等程度差异
    for i in range(key_genes, key_genes + helper_genes):
        if np.random.random() > 0.5:
            # 较弱的上调
            tumor_multiplier = np.random.uniform(1.3, 2.2)
            expression[labels == 1, i] *= tumor_multiplier
            # 更多噪声
            expression[:, i] *= np.random.lognormal(0, 0.5, n_spots)
        else:
            # 较弱的下调
            tumor_multiplier = np.random.uniform(0.5, 0.8)
            expression[labels == 1, i] *= tumor_multiplier
            # 更多噪声
            expression[:, i] *= np.random.lognormal(0, 0.5, n_spots)
    
    # 其余基因（400-1999）：主要是噪声，只有很弱的信号
    for i in range(key_genes + helper_genes, n_genes):
        if np.random.random() < 0.1:  # 只有10%的噪声基因有微弱信号
            tumor_multiplier = np.random.uniform(0.8, 1.2)
            expression[labels == 1, i] *= tumor_multiplier
        # 大量噪声
        expression[:, i] *= np.random.lognormal(0, 0.8, n_spots)
    
    # 确保表达值为正数
    expression = np.maximum(expression, 0.1)
    
    return expression

def generate_single_slide(slide_id: str, n_genes: int = 2000) -> Dict:
    """
    生成单个切片的所有数据
    
    Args:
        slide_id: 切片ID
        n_genes: 基因数量
    
    Returns:
        slide_data: 包含 X, y, xy, slide_id 的字典
    """
    # 随机确定spot数量和肿瘤比例
    n_spots = np.random.randint(2000, 3001)
    tumor_ratio = np.random.uniform(0.4, 0.7)
    
    print(f"生成切片 {slide_id}: {n_spots} spots, 肿瘤比例 {tumor_ratio:.2f}")
    
    # 生成坐标
    coords = generate_visium_coordinates(n_spots)
    
    # 生成肿瘤标签
    labels = generate_tumor_labels(coords, tumor_ratio)
    
    # 生成基因表达
    expression = generate_gene_expression(n_spots, n_genes, labels)
    
    return {
        'X': expression.astype(np.float32),
        'y': labels.astype(np.int32),
        'xy': coords.astype(np.float32),
        'slide_id': slide_id
    }

def generate_gene_names(n_genes: int = 2000) -> List[str]:
    """生成基因名称列表"""
    gene_names = []
    
    # 关键基因（前100个）
    for i in range(100):
        gene_names.append(f"KEY_GENE_{i+1:03d}")
    
    # 辅助基因（100-399）
    for i in range(100, 400):
        gene_names.append(f"HELPER_GENE_{i+1:03d}")
    
    # 其余基因
    for i in range(400, n_genes):
        gene_names.append(f"GENE_{i+1:04d}")
    
    return gene_names

def main():
    parser = argparse.ArgumentParser(description='生成 Visium GNN 模拟数据')
    parser.add_argument('--output_dir', default='synthetic_data', help='输出目录')
    parser.add_argument('--n_genes', type=int, default=2000, help='基因数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始生成模拟数据，输出到: {args.output_dir}")
    
    # 生成基因名称
    gene_names = generate_gene_names(args.n_genes)
    
    # 生成45个切片
    all_slides = []
    for i in range(45):
        slide_id = f"slide_{i+1:02d}"
        slide_data = generate_single_slide(slide_id, args.n_genes)
        all_slides.append(slide_data)
    
    # 分割训练和推理数据
    train_slides = all_slides[:40]
    infer_slides = all_slides[40:]
    
    print(f"\n保存训练数据 (40个切片)...")
    
    # 保存训练数据（单个NPZ文件）
    train_Xs = [slide['X'] for slide in train_slides]
    train_ys = [slide['y'] for slide in train_slides]
    train_xys = [slide['xy'] for slide in train_slides]
    train_slide_ids = [slide['slide_id'] for slide in train_slides]
    
    train_npz_path = os.path.join(args.output_dir, 'train_data.npz')
    np.savez_compressed(
        train_npz_path,
        Xs=np.array(train_Xs, dtype=object),
        ys=np.array(train_ys, dtype=object),
        xys=np.array(train_xys, dtype=object),
        slide_ids=np.array(train_slide_ids, dtype=object),
        gene_names=np.array(gene_names, dtype=object)
    )
    
    print(f"训练数据已保存到: {train_npz_path}")
    
    # 保存推理数据（每个切片单独的NPZ文件）
    print(f"\n保存推理数据 (5个切片)...")
    for i, slide in enumerate(infer_slides):
        infer_npz_path = os.path.join(args.output_dir, f'infer_slide_{i+1}.npz')
        np.savez_compressed(
            infer_npz_path,
            X=slide['X'],
            xy=slide['xy'],
            gene_names=gene_names
        )
        print(f"推理数据已保存到: {infer_npz_path}")
    
    # 生成统计信息
    print(f"\n数据生成完成！统计信息：")
    print(f"- 基因数量: {args.n_genes}")
    print(f"  - 关键基因: 100 (强预测能力)")
    print(f"  - 辅助基因: 300 (中等预测能力)")
    print(f"  - 噪声基因: {args.n_genes - 400}")
    
    total_spots = sum(slide['X'].shape[0] for slide in all_slides)
    total_tumor_spots = sum(slide['y'].sum() for slide in all_slides)
    overall_tumor_ratio = total_tumor_spots / total_spots
    
    print(f"- 总spot数: {total_spots}")
    print(f"- 总肿瘤spot数: {total_tumor_spots}")
    print(f"- 整体肿瘤比例: {overall_tumor_ratio:.3f}")
    
    print(f"- 训练切片: 40个")
    print(f"- 推理切片: 5个")
    
    print(f"\n使用方法:")
    print(f"训练: python train.py --train_npz {train_npz_path} --artifacts_dir artifacts")
    print(f"推理: python infer.py --npz {args.output_dir}/infer_slide_1.npz --artifacts_dir artifacts --out_csv preds.csv")

if __name__ == "__main__":
    main()