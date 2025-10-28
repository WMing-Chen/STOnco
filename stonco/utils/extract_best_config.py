#!/usr/bin/env python3
"""
从 Optuna study.db 文件中提取最佳超参数配置
支持合并三个阶段的最佳参数，生成完整的训练配置

Usage:
python extract_best_config.py --study_dir test_train0819/tuning/gatv2 --out_dir test_train0819/gatv2

说明：
- 本脚本现在会在 out_dir 下生成三个文件：
  1) best_config.json        -> 精简版，结构与训练输出的 meta.json 中的 cfg 一致，可直接供 train.py 使用
  2) best_config_full.json   -> 结构化的完整信息，便于查看阶段信息与来源
  3) best_config.csv         -> 扁平化 CSV 便于对比
"""
import argparse
import json
import pandas as pd
from pathlib import Path
import sqlite3

try:
    import optuna
    optuna_available = True
except ImportError:
    optuna_available = False
    print("Warning: Optuna not available. Will try to extract from SQLite directly.")


def extract_from_sqlite_direct(db_path):
    """直接从 SQLite 数据库提取最佳 trial 信息"""
    try:
        conn = sqlite3.connect(db_path)
        
        # 获取最佳 trial
        query = """
        SELECT trial_id, value
        FROM trials 
        WHERE state = 'COMPLETE' AND value IS NOT NULL
        ORDER BY value DESC 
        LIMIT 1
        """
        cursor = conn.execute(query)
        best_trial_row = cursor.fetchone()
        
        if not best_trial_row:
            print(f"No complete trials found in {db_path}")
            return None, None, None
            
        best_trial_id, best_value = best_trial_row
        
        # 获取该 trial 的参数
        param_query = """
        SELECT param_name, param_value
        FROM trial_params 
        WHERE trial_id = ?
        """
        cursor = conn.execute(param_query, (best_trial_id,))
        params = dict(cursor.fetchall())
        
        conn.close()
        return best_trial_id, best_value, params
        
    except Exception as e:
        print(f"Error extracting from SQLite {db_path}: {e}")
        return None, None, None


def extract_from_optuna_study(db_path, study_name):
    """使用 Optuna API 提取最佳 trial"""
    if not optuna_available:
        return extract_from_sqlite_direct(db_path)
    
    try:
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(storage=storage, study_name=study_name)
        
        if not study.best_trial:
            print(f"No best trial found in study {study_name}")
            return None, None, None
            
        best_trial = study.best_trial
        return best_trial.number, best_trial.value, best_trial.params
        
    except Exception as e:
        print(f"Error loading Optuna study from {db_path}: {e}")
        return extract_from_sqlite_direct(db_path)


def get_base_config():
    """获取基础配置（不变的参数）"""
    return {
        "model": "gatv2",
        "knn_k": 6,
        "gaussian_sigma_factor": 1.0,
        "pca_dim": 64,
        "use_pca": False,
        "use_domain_adv": True,
        "domain_lambda": 0.3,
        "early_patience": 30,
        "heads": 4,  # GATv2 默认
    }


def format_optuna_param_value(param_name, param_value):
    """格式化 Optuna 参数值（处理类别参数的索引映射）"""
    # 根据超参数优化方案中的定义进行映射
    if param_name == "lap_pe_dim":
        # lap_pe_dim 类别: {8, 12, 16, 20}
        choices = [8, 12, 16, 20]
        if isinstance(param_value, (int, float)) and 0 <= param_value < len(choices):
            return choices[int(param_value)]
        return param_value
    elif param_name == "hidden":
        # hidden 类别: {64, 96, 128, 192, 256}
        choices = [64, 96, 128, 192, 256]
        if isinstance(param_value, (int, float)) and 0 <= param_value < len(choices):
            return choices[int(param_value)]
        return param_value
    elif param_name == "batch_size_graphs":
        # batch_size_graphs 类别: {1, 2, 4}
        choices = [1, 2, 4]
        if isinstance(param_value, (int, float)) and 0 <= param_value < len(choices):
            return choices[int(param_value)]
        return param_value
    elif param_name in ["concat_lap_pe", "lap_pe_use_gaussian"]:
        # 布尔类型参数
        return bool(int(param_value)) if isinstance(param_value, (int, float, str)) else param_value
    else:
        return param_value


def extract_best_configs(study_dir):
    """从三个阶段的 study.db 中提取最佳配置"""
    study_dir = Path(study_dir)
    
    stages_config = {}
    stage_info = {}
    
    for stage in ["stage1", "stage2", "stage3"]:
        db_path = study_dir / stage / "study.db"
        if not db_path.exists():
            print(f"Warning: {db_path} not found, skipping {stage}")
            continue
            
        study_name = f"gatv2_{stage}"  # 根据训练脚本中的命名规则
        trial_id, best_value, params = extract_from_optuna_study(db_path, study_name)
        
        if params is None:
            print(f"Warning: No valid params found for {stage}")
            continue
            
        # 格式化参数值
        formatted_params = {}
        for k, v in params.items():
            formatted_params[k] = format_optuna_param_value(k, v)
            
        stages_config[stage] = formatted_params
        stage_info[stage] = {
            "best_trial_id": trial_id,
            "best_value": best_value,
            "param_count": len(formatted_params)
        }
        
        print(f"{stage}: trial_id={trial_id}, value={best_value:.6f}, params={len(formatted_params)}")
    
    return stages_config, stage_info


def create_combined_config(stages_config, stage_info):
    """合并各阶段配置为完整配置"""
    base_config = get_base_config()
    
    # 按阶段合并参数
    combined = dict(base_config)
    stage_params = {
        "stage1_optimal": {},
        "stage2_optimal": {},
        "stage3_optimal": {}
    }
    
    # 根据超参数优化方案的阶段定义分类参数
    stage1_params = {"lr", "weight_decay", "epochs", "batch_size_graphs"}
    stage2_params = {"hidden", "num_layers", "dropout"}
    stage3_params = {"lap_pe_dim", "lap_pe_use_gaussian", "concat_lap_pe"}
    
    for stage, params in stages_config.items():
        for param_name, param_value in params.items():
            combined[param_name] = param_value
            
            # 分类到对应阶段
            if param_name in stage1_params:
                stage_params["stage1_optimal"][param_name] = param_value
            elif param_name in stage2_params:
                stage_params["stage2_optimal"][param_name] = param_value
            elif param_name in stage3_params:
                stage_params["stage3_optimal"][param_name] = param_value
    
    # 构造完整的结构化配置
    full_config = {
        "model_info": {
            "model_name": "gatv2",
            "training_mode": "no_pca",
            "device": "cuda",  
            "optimization_stages": "_".join(sorted(stages_config.keys())),
            "note": "Best configuration extracted from Optuna study databases"
        },
        "stage1_optimal": {
            "note": "Core training hyperparameters (convergence & generalization priority)",
            **stage_params["stage1_optimal"]
        },
        "stage2_optimal": {
            "note": "Network architecture (capacity & regularization)",
            **stage_params["stage2_optimal"]
        },
        "stage3_optimal": {
            "note": "Graph positional encoding fine-tuning",
            **stage_params["stage3_optimal"]
        },
        "graph_construction": {
            "knn_k": combined.get("knn_k", 6),
            "gaussian_sigma_factor": combined.get("gaussian_sigma_factor", 1.0)
        },
        "preprocessing": {
            "pca_dim": combined.get("pca_dim", 64),
            "use_pca": combined.get("use_pca", False)
        },
        "domain_adaptation": {
            "use_domain_adv": combined.get("use_domain_adv", True),
            "domain_lambda": combined.get("domain_lambda", 0.3)
        },
        "extraction_info": {
            "stages_found": list(stages_config.keys()),
            "stage_info": stage_info
        },
        "complete_config": combined
    }
    
    # 将 stage3 的最佳 trial 信息添加到 stage3_optimal
    if "stage3" in stage_info:
        full_config["stage3_optimal"]["best_trial_id"] = stage_info["stage3"]["best_trial_id"]
        full_config["stage3_optimal"]["best_value"] = stage_info["stage3"]["best_value"]
    
    return full_config, combined


def save_configs(full_config, combined_config, out_dir):
    """保存配置到 JSON 和 CSV 文件
    - best_config.json: 精简版，可被 train.py --config_json 直接读取（形如 {"cfg": {...}}）
    - best_config_full.json: 结构化完整信息
    - best_config.csv: 扁平化 CSV
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) 保存精简的、可直接用于训练的配置（与 meta.json 的 cfg 一致）
    train_ready = {"cfg": combined_config}
    best_simple_path = out_dir / "best_config.json"
    with open(best_simple_path, 'w', encoding='utf-8') as f:
        json.dump(train_ready, f, indent=2, ensure_ascii=False)
    print(f"Saved train-ready config to: {best_simple_path}")
    
    # 2) 保存完整的结构化配置到 JSON
    full_json_path = out_dir / "best_config_full.json"
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)
    print(f"Saved structured config to: {full_json_path}")
    
    # 3) 保存扁平化配置到 CSV（便于对比和使用）
    csv_data = []
    for param, value in combined_config.items():
        # 确定参数来源阶段
        stage1_params = {"lr", "weight_decay", "epochs", "batch_size_graphs"}
        stage2_params = {"hidden", "num_layers", "dropout"}
        stage3_params = {"lap_pe_dim", "lap_pe_use_gaussian", "concat_lap_pe"}
        
        if param in stage1_params:
            stage = "stage1"
        elif param in stage2_params:
            stage = "stage2"
        elif param in stage3_params:
            stage = "stage3"
        else:
            stage = "base"
            
        csv_data.append({
            "param": param,
            "value": value,
            "stage": stage,
            "note": ""
        })
    
    csv_path = out_dir / "best_config.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Saved flat config to: {csv_path}")
    
    return best_simple_path, full_json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Extract best hyperparameters from Optuna studies")
    parser.add_argument("--study_dir", required=True, help="Directory containing stage1/stage2/stage3 subdirs with study.db")
    parser.add_argument("--out_dir", required=True, help="Output directory for best_config.json and best_config.csv")
    
    args = parser.parse_args()
    
    print(f"Extracting best configs from: {args.study_dir}")
    print(f"Output directory: {args.out_dir}")
    
    # 提取各阶段最佳配置
    stages_config, stage_info = extract_best_configs(args.study_dir)
    
    if not stages_config:
        print("No valid configurations found in any stage!")
        return
    
    # 合并配置
    full_config, combined_config = create_combined_config(stages_config, stage_info)
    
    # 保存配置
    simple_json_path, full_json_path, csv_path = save_configs(full_config, combined_config, args.out_dir)
    
    print("\n=== Extraction Summary ===")
    print(f"Stages processed: {list(stages_config.keys())}")
    print(f"Total unique parameters: {len(combined_config)}")
    print(f"Output files: {simple_json_path}, {full_json_path}, {csv_path}")
    
    # 显示关键参数摘要
    print("\n=== Key Parameters ===")
    key_params = ["lr", "weight_decay", "hidden", "num_layers", "dropout", "lap_pe_dim", "concat_lap_pe"]
    for param in key_params:
        if param in combined_config:
            print(f"  {param}: {combined_config[param]}")


if __name__ == "__main__":
    main()