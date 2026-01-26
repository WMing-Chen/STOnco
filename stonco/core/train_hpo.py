from .train import prepare_graphs, train_and_validate

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
import gc

# HPO相关导入
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner
    optuna_available = True
except ImportError:
    optuna_available = False



def run_hyperparameter_optimization(args, cfg, device):
    if not optuna_available:
        raise RuntimeError('Optuna 未安装，无法进行超参数优化。请先安装 optuna。')

    # HPO阶段强制禁用PCA（device 由 main 统一处理，支持 GPU）
    cfg = dict(cfg)
    cfg['use_pca'] = False

    # 准备数据（不保存预处理器）
    train_graphs, val_graphs, in_dim, n_domains_slide, n_domains_cancer = prepare_graphs(args, cfg, save_preprocessor_dir=None)

    # 结果目录
    stage = args.tune
    tune_dir = Path(args.artifacts_dir).parent / 'tuning' / stage
    tune_dir.mkdir(parents=True, exist_ok=True)

    # 构建study
    sampler = TPESampler(seed=42)
    pruner = SuccessiveHalvingPruner()
    study_kwargs = dict(direction='maximize', sampler=sampler, pruner=pruner)
    if args.storage:
        study = optuna.create_study(storage=args.storage, study_name=(args.study_name or f"{cfg['model']}_{stage}"), load_if_exists=True, **study_kwargs)
    else:
        # 默认将 study.db 写入 tune_dir 目录
        default_storage = f"sqlite:///{tune_dir}/study.db"
        study = optuna.create_study(storage=default_storage, study_name=(args.study_name or f"{cfg['model']}_{stage}"), load_if_exists=True, **study_kwargs)

    def objective(trial: 'optuna.trial.Trial'):
        # 每个trial从base cfg构造
        trial_cfg = _get_stage_search_space(stage, trial, cfg)
        # 训练轮数可在stage1较小加速（如未显式指定）
        # 保持用户若传入epochs则尊重
        seed = 1000 + trial.number
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            def report_cb(epoch, metrics):
                # 以验证准确率为唯一优化目标
                trial.report(metrics.get('accuracy', float('nan')), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            best, hist, _ = train_and_validate(train_graphs, val_graphs, in_dim, n_domains_slide, n_domains_cancer, trial_cfg, device, num_workers=args.num_workers, report_cb=report_cb)
            return best.get('accuracy', float('nan'))
        finally:
            # 显式清理：删除本trial局部变量并清空CUDA缓存
            try:
                del best
            except Exception:
                pass
            try:
                del hist
            except Exception:
                pass
            try:
                del trial_cfg
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            try:
                gc.collect()
            except Exception:
                pass

    study.optimize(objective, n_trials=args.n_trials)

    # 保存trials.csv
    recs = []
    for t in study.trials:
        rec = {**t.params, 'value': t.value, 'state': str(t.state), 'number': t.number}
        recs.append(rec)
    pd.DataFrame(recs).to_csv(tune_dir / 'trials.csv', index=False)

    # 保存最优配置
    best_params = study.best_trial.params if len(study.best_trial.params)>0 else {}
    best_cfg = dict(cfg); best_cfg.update(best_params)
    save_json(best_cfg, str(tune_dir / f'best_config_{stage}.json'))
    print(f'HPO完成：最佳准确率={study.best_value:.4f}，配置已保存到 {tune_dir}')


def run_rescore_topk(args, cfg, device):
    if not optuna_available:
        raise RuntimeError('Optuna 未安装，无法进行多种子复评。')

    # 复评阶段强制禁用PCA
    cfg = dict(cfg)
    cfg['use_pca'] = False

    # 目录（与 HPO 一致）
    stage = args.tune or 'stage3'
    tune_dir = Path(args.artifacts_dir).parent / 'tuning' / stage
    tune_dir.mkdir(parents=True, exist_ok=True)
     
    # 读取study
    # 读取study（支持默认 storage 与 study_name）
    storage = args.storage if args.storage else f"sqlite:///{tune_dir}/study.db"
    study_name = args.study_name if args.study_name else f"{cfg['model']}_{stage}"
    study = optuna.load_study(storage=storage, study_name=study_name)

    # Top-K trials
    topk = args.rescore_topk
    sorted_trials = [t for t in study.trials if t.value is not None and t.state==optuna.trial.TrialState.COMPLETE]
    sorted_trials.sort(key=lambda t: t.value, reverse=True)
    chosen = sorted_trials[:topk]

    # 准备数据
    train_graphs, val_graphs, in_dim, n_domains_slide, n_domains_cancer = prepare_graphs(args, cfg, save_preprocessor_dir=None)

    # 保存目录已在前面创建，以下逻辑保留
    out_json = tune_dir / 'topk_rescore.json'
    seeds = [int(s.strip()) for s in str(args.seeds).split(',') if len(s.strip())>0]
    results = []
    for rank, t in enumerate(chosen, start=1):
        params = dict(t.params)
        cfg_i = dict(cfg); cfg_i.update(params)
        per_seed = []
        for sd in seeds:
            np.random.seed(sd); random.seed(sd); torch.manual_seed(sd)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sd)
            best, _, _ = train_and_validate(train_graphs, val_graphs, in_dim, n_domains_slide, n_domains_cancer, cfg_i, device, num_workers=args.num_workers)
            per_seed.append({'seed': sd, 'accuracy': best.get('accuracy', float('nan')), 'auroc': best.get('auroc', float('nan')), 'macro_f1': best.get('macro_f1', float('nan'))})
        # 汇总
        acc_vals = [r['accuracy'] for r in per_seed if not np.isnan(r['accuracy'])]
        mean_acc = float(np.mean(acc_vals)) if acc_vals else float('nan')
        std_acc = float(np.std(acc_vals)) if acc_vals else float('nan')
        results.append({'rank': rank, 'trial_number': t.number, 'params': params, 'per_seed': per_seed, 'mean_accuracy': mean_acc, 'std_accuracy': std_acc})

    # 保存
    out_json = tune_dir / 'topk_rescore.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('已保存多种子复评结果到', out_json)

    # 选择均值最优并保存
    if results:
        best_item = max(results, key=lambda r: (r['mean_accuracy'] if not np.isnan(r['mean_accuracy']) else -1))
        best_cfg = dict(cfg); best_cfg.update(best_item['params'])
        save_json(best_cfg, str(tune_dir / 'best_config_rescored.json'))
        print('已保存复评后的最佳配置到', tune_dir / 'best_config_rescored.json')


def _get_stage_search_space(stage, trial, base_cfg):
    cfg = dict(base_cfg)
    if stage == 'stage1':
        cfg['lr'] = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        wd_zero = trial.suggest_categorical('wd_zero', [0, 1])
        if wd_zero == 1:
            cfg['weight_decay'] = 0.0
        else:
            cfg['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        cfg['epochs'] = trial.suggest_int('epochs', 80, 200)
        cfg['batch_size_graphs'] = trial.suggest_categorical('batch_size_graphs', [1, 2, 4])
    elif stage == 'stage2':
        cfg['hidden'] = trial.suggest_categorical('hidden', [64, 128, 192, 256])
        cfg['num_layers'] = trial.suggest_int('num_layers', 2, 5)
        cfg['dropout'] = trial.suggest_float('dropout', 0.1, 0.6)
        if cfg['model'] == 'gatv2':
            cfg['heads'] = trial.suggest_categorical('heads', [2, 4, 6, 8])
    elif stage == 'stage3':
        cfg['concat_lap_pe'] = trial.suggest_categorical('concat_lap_pe', [0, 1]) == 1
        cfg['lap_pe_use_gaussian'] = trial.suggest_categorical('lap_pe_use_gaussian', [0, 1]) == 1
        cfg['lap_pe_dim'] = trial.suggest_categorical('lap_pe_dim', [8, 12, 16, 20])
    return cfg


def run_multi_stage_hpo(args, cfg, device):
    """统一的多阶段HPO函数，按顺序执行stage1->stage2->stage3，完成后进行选择性复评并合并最佳配置"""
    if not optuna_available:
        raise RuntimeError('Optuna 未安装，无法进行超参数优化。请先安装 optuna。')

    # HPO阶段强制禁用PCA（device 由 main 统一处理，支持 GPU）
    cfg = dict(cfg)
    cfg['use_pca'] = False

    # 总进度条：共3个stage（stage1, 2, 3）+ 可选复评阶段数
    rescore_stages = [int(s.strip()) for s in args.rescore_stages.split(',') if s.strip() and s.strip() in {'1','2','3'}]
    total_steps = 3 + len(rescore_stages)  # 3个调参阶段 + 复评阶段数
    
    main_progress = tqdm(total=total_steps, desc="多阶段HPO流水线", position=0, leave=True)
    
    # tuning根目录
    tune_base_dir = Path(args.artifacts_dir).parent / 'tuning'
    tune_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储每个阶段的最佳配置
    stage_best_configs = {}
    
    # 准备数据一次，因为stage1和stage2不需要重新构建图
    base_train_graphs, base_val_graphs, in_dim, n_domains_slide, n_domains_cancer = prepare_graphs(args, cfg, save_preprocessor_dir=None)
    
    stages = ['stage1', 'stage2', 'stage3']
    
    for stage in stages:
        main_progress.set_description(f"执行 {stage} HPO")
        
        # stage目录
        stage_dir = tune_base_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果是stage3，需要重新构建图（lap PE可能变化）
        if stage == 'stage3':
            train_graphs, val_graphs = base_train_graphs, base_val_graphs
        else:
            train_graphs, val_graphs = base_train_graphs, base_val_graphs
            
        # 构建study
        sampler = TPESampler(seed=42)
        pruner = SuccessiveHalvingPruner()
        study_kwargs = dict(direction='maximize', sampler=sampler, pruner=pruner)
        
        storage = args.storage if args.storage else f"sqlite:///{stage_dir}/study.db"
        study_name = args.study_name if args.study_name else f"{cfg['model']}_{stage}"
        
        study = optuna.create_study(storage=storage, study_name=study_name, load_if_exists=True, **study_kwargs)

        def objective(trial: 'optuna.trial.Trial'):
            try:
                # 每个trial从base cfg构造，累积之前stage的最佳参数
                trial_cfg = dict(cfg)
                # 累积之前阶段的最佳参数
                for prev_stage in ['stage1', 'stage2']:
                    if prev_stage in stage_best_configs and prev_stage != stage:
                        trial_cfg.update(stage_best_configs[prev_stage])
                
                # 应用当前stage的搜索空间
                trial_cfg = _get_stage_search_space(stage, trial, trial_cfg)
                
                # 如果是stage3，可能需要重新构建图
                if stage == 'stage3':
                    # stage3可能改变lap PE参数，需要重新构建图
                    current_train_graphs, current_val_graphs, in_dim_current, n_domains_slide_current, n_domains_cancer_current = prepare_graphs(args, trial_cfg, save_preprocessor_dir=None)
                else:
                    current_train_graphs, current_val_graphs = train_graphs, val_graphs
                    in_dim_current = in_dim
                    n_domains_slide_current = n_domains_slide
                    n_domains_cancer_current = n_domains_cancer
                
                seed = 1000 + trial.number
                np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                def report_cb(epoch, metrics):
                    trial.report(metrics.get('accuracy', float('nan')), epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                best, hist, _ = train_and_validate(current_train_graphs, current_val_graphs, in_dim_current, n_domains_slide_current, n_domains_cancer_current, trial_cfg, device, num_workers=args.num_workers, report_cb=report_cb)
                return best.get('accuracy', float('nan'))
            finally:
                # 显式清理：删除本trial局部变量并清空CUDA缓存
                for _name in ['best', 'hist', 'trial_cfg', 'current_train_graphs', 'current_val_graphs', 'in_dim_current', 'n_domains_slide_current', 'n_domains_cancer_current']:
                    try:
                        del locals()[_name]
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                try:
                    gc.collect()
                except Exception:
                    pass

        # 使用子进度条显示当前stage的试验进度
        with tqdm(total=args.n_trials, desc=f"{stage} 试验", position=1, leave=False) as trial_progress:
            def callback(study, trial):
                trial_progress.update(1)
                trial_progress.set_postfix({"best": f"{study.best_value:.4f}" if study.best_value else "N/A"})
            
            study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])

        # 保存当前stage结果
        recs = []
        for t in study.trials:
            rec = {**t.params, 'value': t.value, 'state': str(t.state), 'number': t.number}
            recs.append(rec)
        pd.DataFrame(recs).to_csv(stage_dir / 'trials.csv', index=False)

        # 保存最优配置
        best_params = study.best_trial.params if study.best_trial and len(study.best_trial.params) > 0 else {}
        stage_cfg = dict(cfg)
        # 累积之前阶段最佳参数
        for prev_stage in ['stage1', 'stage2']:
            if prev_stage in stage_best_configs and prev_stage != stage:
                stage_cfg.update(stage_best_configs[prev_stage])
        stage_cfg.update(best_params)
        
        save_json(stage_cfg, str(stage_dir / f'best_config_{stage}.json'))
        stage_best_configs[stage] = best_params  # 只存储当前stage的参数
        
        print(f'{stage} HPO完成：最佳准确率={study.best_value:.4f}')
        main_progress.update(1)

    # 复评阶段
    if rescore_stages and args.rescore_topk:
        for stage_num in rescore_stages:
            stage = f'stage{stage_num}'
            main_progress.set_description(f"复评 {stage}")
            
            # 执行复评
            stage_dir = tune_base_dir / stage
            storage = f"sqlite:///{stage_dir}/study.db"
            study_name = f"{cfg['model']}_{stage}"
            
            try:
                study = optuna.load_study(storage=storage, study_name=study_name)
                
                # Top-K trials
                topk = args.rescore_topk
                sorted_trials = [t for t in study.trials if t.value is not None and t.state==optuna.trial.TrialState.COMPLETE]
                sorted_trials.sort(key=lambda t: t.value, reverse=True)
                chosen = sorted_trials[:topk]
                
                seeds = [int(s.strip()) for s in str(args.seeds).split(',') if len(s.strip())>0]
                results = []
                
                for rank, t in enumerate(chosen, start=1):
                    params = dict(t.params)
                    # 构建完整配置（累积所有前序阶段参数）
                    full_cfg = dict(cfg)
                    for prev_stage in ['stage1', 'stage2', 'stage3']:
                        if prev_stage in stage_best_configs:
                            full_cfg.update(stage_best_configs[prev_stage])
                    full_cfg.update(params)
                    
                    per_seed = []
                    for sd in seeds:
                        np.random.seed(sd); random.seed(sd); torch.manual_seed(sd)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(sd)
                        
                        # 重新构建图（如果stage3改变了lap PE参数）
                        if stage == 'stage3':
                            rescore_train_graphs, rescore_val_graphs, in_dim_current, n_domains_slide_current, n_domains_cancer_current = prepare_graphs(args, full_cfg, save_preprocessor_dir=None)
                        else:
                            rescore_train_graphs, rescore_val_graphs = base_train_graphs, base_val_graphs
                            in_dim_current = in_dim
                            n_domains_slide_current = n_domains_slide
                            n_domains_cancer_current = n_domains_cancer
                            
                        best, _, _ = train_and_validate(rescore_train_graphs, rescore_val_graphs, in_dim_current, n_domains_slide_current, n_domains_cancer_current, full_cfg, device, num_workers=args.num_workers)
                        per_seed.append({'seed': sd, 'accuracy': best.get('accuracy', float('nan')), 'auroc': best.get('auroc', float('nan')), 'macro_f1': best.get('macro_f1', float('nan'))})
                        
                        # 清理每个 seed 训练的缓存，避免复评阶段显存堆积
                        try:
                            del best
                        except Exception:
                            pass
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        try:
                            gc.collect()
                        except Exception:
                            pass
                    
                    # 汇总
                    acc_vals = [r['accuracy'] for r in per_seed if not np.isnan(r['accuracy'])]
                    mean_acc = float(np.mean(acc_vals)) if acc_vals else float('nan')
                    std_acc = float(np.std(acc_vals)) if acc_vals else float('nan')
                    results.append({'rank': rank, 'trial_number': t.number, 'params': params, 'per_seed': per_seed, 'mean_accuracy': mean_acc, 'std_accuracy': std_acc})

                # 保存复评结果
                out_json = stage_dir / 'topk_rescore.json'
                with open(out_json, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # 选择均值最优并更新stage配置
                if results:
                    best_item = max(results, key=lambda r: (r['mean_accuracy'] if not np.isnan(r['mean_accuracy']) else -1))
                    rescored_cfg = dict(cfg)
                    # 累积所有前序阶段参数
                    for prev_stage in ['stage1', 'stage2', 'stage3']:
                        if prev_stage in stage_best_configs:
                            rescored_cfg.update(stage_best_configs[prev_stage])
                    rescored_cfg.update(best_item['params'])
                    save_json(rescored_cfg, str(stage_dir / 'best_config_rescored.json'))
                    
                    # 更新该stage的最佳参数为复评后的结果
                    stage_best_configs[stage] = best_item['params']
                    print(f'{stage} 复评完成，复评后最佳配置已保存')
                    
            except Exception as e:
                print(f'警告：{stage} 复评失败: {e}')
            
            main_progress.update(1)
    
    # 合并所有阶段最佳配置并保存到 tuning/best_config.json
    main_progress.set_description("合并最佳配置")
    final_cfg = dict(cfg)
    for stage in ['stage1', 'stage2', 'stage3']:
        if stage in stage_best_configs:
            final_cfg.update(stage_best_configs[stage])
    
    save_json(final_cfg, str(tune_base_dir / 'best_config.json'))
    print(f'所有阶段HPO完成，最终最佳配置已保存到 {tune_base_dir / "best_config.json"}')
    
    main_progress.close()



def run_rescore_multiple_stages(args, cfg, device):
    """支持多阶段复评的函数"""
    if not optuna_available:
        raise RuntimeError('Optuna 未安装，无法进行多种子复评。')

    # 复评阶段强制禁用PCA
    cfg = dict(cfg)
    cfg['use_pca'] = False

    # 解析复评阶段
    stages_to_rescore = [int(s.strip()) for s in args.rescore_stages.split(',') if s.strip() and s.strip() in '123']
    if not stages_to_rescore:
        stages_to_rescore = [1]  # 默认只复评stage1
    
    tune_base_dir = Path(args.artifacts_dir).parent / 'tuning'
    
    # 准备数据
    train_graphs, val_graphs, in_dim, n_domains_slide, n_domains_cancer = prepare_graphs(args, cfg, save_preprocessor_dir=None)
    
    for stage_num in stages_to_rescore:
        stage = f'stage{stage_num}'
        print(f'开始复评 {stage}...')
        
        stage_dir = tune_base_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取study
        storage = args.storage if args.storage else f"sqlite:///{stage_dir}/study.db"
        study_name = args.study_name if args.study_name else f"{cfg['model']}_{stage}"
        
        try:
            study = optuna.load_study(storage=storage, study_name=study_name)
        except Exception as e:
            print(f'无法加载 {stage} 的study: {e}')
            continue

        # Top-K trials
        topk = args.rescore_topk
        sorted_trials = [t for t in study.trials if t.value is not None and t.state==optuna.trial.TrialState.COMPLETE]
        sorted_trials.sort(key=lambda t: t.value, reverse=True)
        chosen = sorted_trials[:topk]

        seeds = [int(s.strip()) for s in str(args.seeds).split(',') if len(s.strip())>0]
        results = []
        for rank, t in enumerate(chosen, start=1):
            params = dict(t.params)
            cfg_i = dict(cfg); cfg_i.update(params)
            per_seed = []
            for sd in seeds:
                np.random.seed(sd); random.seed(sd); torch.manual_seed(sd)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sd)
                    
                # 如果是stage3，可能需要重新构建图
                if stage == 'stage3':
                    rescore_train_graphs, rescore_val_graphs, in_dim_current, n_domains_slide_current, n_domains_cancer_current = prepare_graphs(args, cfg_i, save_preprocessor_dir=None)
                else:
                    rescore_train_graphs, rescore_val_graphs = train_graphs, val_graphs
                    in_dim_current = in_dim
                    n_domains_slide_current = n_domains_slide
                    n_domains_cancer_current = n_domains_cancer
                    
                best, _, _ = train_and_validate(rescore_train_graphs, rescore_val_graphs, in_dim_current, n_domains_slide_current, n_domains_cancer_current, cfg_i, device, num_workers=args.num_workers)
                per_seed.append({'seed': sd, 'accuracy': best.get('accuracy', float('nan')), 'auroc': best.get('auroc', float('nan')), 'macro_f1': best.get('macro_f1', float('nan'))})
            # 汇总
            acc_vals = [r['accuracy'] for r in per_seed if not np.isnan(r['accuracy'])]
            mean_acc = float(np.mean(acc_vals)) if acc_vals else float('nan')
            std_acc = float(np.std(acc_vals)) if acc_vals else float('nan')
            results.append({'rank': rank, 'trial_number': t.number, 'params': params, 'per_seed': per_seed, 'mean_accuracy': mean_acc, 'std_accuracy': std_acc})

        # 保存
        out_json = stage_dir / 'topk_rescore.json'
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'{stage} 复评结果已保存到', out_json)

        # 选择均值最优并保存
        if results:
            best_item = max(results, key=lambda r: (r['mean_accuracy'] if not np.isnan(r['mean_accuracy']) else -1))
            best_cfg = dict(cfg); best_cfg.update(best_item['params'])
            save_json(best_cfg, str(stage_dir / 'best_config_rescored.json'))
            print(f'{stage} 复评后的最佳配置已保存到', stage_dir / 'best_config_rescored.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', required=True)
    parser.add_argument('--artifacts_dir', default='artifacts')
    
    # 现有可选参数
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--early_patience', type=int, default=None, help='早停耐心值，<=0 表示关闭早停')
    parser.add_argument('--batch_size_graphs', type=int, default=None)
    parser.add_argument('--disable_domain_adv', action='store_true', help='关闭域自适应（DomainAdversarial）训练')
    parser.add_argument('--model', choices=['gatv2', 'sage', 'gcn'], default=None, help='选择GNN主干')
    parser.add_argument('--heads', type=int, default=None, help='GATv2的多头数（仅对gatv2有效）')
    parser.add_argument('--concat_lap_pe', type=int, choices=[0,1], default=None, help='是否将lapPE拼接至节点特征（1/0）')
    parser.add_argument('--lap_pe_use_gaussian', type=int, choices=[0,1], default=None, help='lapPE是否使用高斯边权（1/0）')
    parser.add_argument('--lap_pe_dim', type=int, default=None, help='lapPE维度（>0表示启用）')
    parser.add_argument('--num_threads', type=int, default=None, help='设置PyTorch计算线程数（CPU模式下限制核心占用）')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader数据加载工作进程数')
    parser.add_argument('--use_pca', type=int, choices=[0, 1], default=None, help='是否使用PCA（1/0）')
    
    # 新增：网络结构与优化器超参数
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--hidden', type=int, default=None, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=None, help='GNN层数')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout比例')
    
    # 新增：设备控制
    parser.add_argument('--device', default=None, help='指定设备（cpu/cuda）,不指定则自动检测')
    
    # 新增：HPO 调参相关参数
    parser.add_argument('--tune', choices=['stage1', 'stage2', 'stage3', 'all'], default=None, 
                       help='开启超参数优化模式，指定优化阶段或all执行完整3阶段流水线')
    parser.add_argument('--n_trials', type=int, default=30, help='每阶段试验次数')
    parser.add_argument('--study_name', default=None, help='Optuna study 名称')
    parser.add_argument('--storage', default=None, help='Optuna storage 路径（SQLite）')
    
    # 新增：多种子复评相关参数
    parser.add_argument('--rescore_topk', type=int, default=None, 
                       help='对指定 study 的 Top-K 配置进行多种子复评')
    parser.add_argument('--rescore_stages', default='1', 
                       help='复评阶段（逗号分隔，如1,2,3）默认只复评stage1')
    parser.add_argument('--seeds', default='42,2023,2024', 
                       help='复评使用的随机种子列表（逗号分隔）')

    # 新增：癌种分层与K折
    parser.add_argument('--stratify_by_cancer', action='store_true', default=True, help='启用癌种分层划分：按比例分配验证集且每癌种保底1张（n=1除外）')
    parser.add_argument('--no_stratify_by_cancer', action='store_false', dest='stratify_by_cancer', help='关闭癌种分层，使用简单划分（最后1张为验证）')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例（默认0.2，按癌种分配且保底1张）')
    parser.add_argument('--kfold_cancer', type=int, default=None, help='基于癌种的K折（按比例划分验证集并随机组合）')
    parser.add_argument('--split_seed', type=int, default=42, help='分层/交叉验证随机种子')
    parser.add_argument('--split_test_only', action='store_true', help='仅测试划分逻辑，不进行训练，打印每折的统计信息')

    # 新增：双域对抗控制与权重（与 train.py 对齐）
    parser.add_argument('--use_domain_adv_slide', type=int, choices=[0,1], default=None, help='启用/关闭切片域对抗（1/0）')
    parser.add_argument('--use_domain_adv_cancer', type=int, choices=[0,1], default=None, help='启用/关闭癌种域对抗（1/0）')
    parser.add_argument('--lambda_slide', type=float, default=None, help='切片域对抗损失权重')
    parser.add_argument('--lambda_cancer', type=float, default=None, help='癌种域对抗损失权重')
    # 新增：GRL beta schedule（与 train.py 对齐）
    parser.add_argument('--grl_beta_mode', choices=['dann', 'constant'], default=None, help='GRL beta 模式：dann(从0平滑涨到target)/constant(全程恒定=target)')
    parser.add_argument('--grl_beta_slide_target', type=float, default=None, help='切片域 GRL 目标强度（dann/constant），默认1.0')
    parser.add_argument('--grl_beta_cancer_target', type=float, default=None, help='癌种域 GRL 目标强度（dann/constant），默认0.5')
    parser.add_argument('--grl_beta_gamma', type=float, default=None, help='GRL DANN schedule gamma，默认10')
    # 新增：HVG控制
    parser.add_argument('--n_hvg', default='all', help="高变基因数量，或'all'使用全部基因（默认'all'）")
    
    args = parser.parse_args()

    cfg = {'pca_dim':64, 'lap_pe_dim':16, 'knn_k':6, 'gaussian_sigma_factor':1.0, 'hidden':128, 'num_layers':3, 'dropout':0.3, 'model':'gatv2', 'heads':4, 'domain_lambda':0.3, 'lr':1e-3, 'weight_decay':1e-4, 'epochs':100, 'batch_size_graphs':2, 'early_patience':30,
	           # 控制项
	           'use_pca': False,
	           'concat_lap_pe': True,
	           'lap_pe_use_gaussian': False,
	           # 双域默认配置（与 train.py 对齐）
		           'use_domain_adv_slide': True,   # 默认开启（batch/slide 域）
		           'use_domain_adv_cancer': True,  # 默认开启
		           'lambda_slide': None,           # 若None，将回退到 domain_lambda
		           'lambda_cancer': None,          # 若None，将回退到 domain_lambda
		           # beta（GRL 对抗强度，与 train.py 对齐）
		           'grl_beta_mode': 'dann',
		           'grl_beta_slide_target': 1.0,
		           'grl_beta_cancer_target': 0.5,
		           'grl_beta_gamma': 10.0,
		           # HVG控制（保持与 train.py 一致）
		           'n_hvg': 'all'
		           }

    # 覆盖配置以支持快速实验和HPO
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.early_patience is not None:
        cfg['early_patience'] = args.early_patience
    if args.batch_size_graphs is not None:
        cfg['batch_size_graphs'] = args.batch_size_graphs
    if args.disable_domain_adv:
        cfg['use_domain_adv_slide'] = False
        cfg['use_domain_adv_cancer'] = False
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
    
    # 新增：覆盖网络结构与优化器超参数
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
    if getattr(args, 'use_domain_adv_slide', None) is not None:
        cfg['use_domain_adv_slide'] = bool(args.use_domain_adv_slide)
    if getattr(args, 'use_domain_adv_cancer', None) is not None:
        cfg['use_domain_adv_cancer'] = bool(args.use_domain_adv_cancer)
    if getattr(args, 'lambda_slide', None) is not None:
        cfg['lambda_slide'] = float(args.lambda_slide)
    if getattr(args, 'lambda_cancer', None) is not None:
        cfg['lambda_cancer'] = float(args.lambda_cancer)
    if getattr(args, 'grl_beta_mode', None) is not None:
        cfg['grl_beta_mode'] = str(args.grl_beta_mode)
    if getattr(args, 'grl_beta_slide_target', None) is not None:
        cfg['grl_beta_slide_target'] = float(args.grl_beta_slide_target)
    if getattr(args, 'grl_beta_cancer_target', None) is not None:
        cfg['grl_beta_cancer_target'] = float(args.grl_beta_cancer_target)
    if getattr(args, 'grl_beta_gamma', None) is not None:
        cfg['grl_beta_gamma'] = float(args.grl_beta_gamma)

    # 默认值填充（新字段，与 train.py 对齐）
    if cfg.get('use_domain_adv_slide', None) is None:
        cfg['use_domain_adv_slide'] = True
    cfg['use_domain_adv_slide'] = bool(cfg['use_domain_adv_slide'])
    cfg['use_domain_adv_cancer'] = bool(cfg.get('use_domain_adv_cancer', True))
    if cfg.get('lambda_slide', None) is None:
        cfg['lambda_slide'] = float(cfg.get('domain_lambda', 0.3))
    if cfg.get('lambda_cancer', None) is None:
        cfg['lambda_cancer'] = float(cfg.get('domain_lambda', 0.3))
    if cfg.get('grl_beta_mode', None) is None:
        cfg['grl_beta_mode'] = 'dann'
    if cfg.get('grl_beta_slide_target', None) is None:
        cfg['grl_beta_slide_target'] = 1.0
    if cfg.get('grl_beta_cancer_target', None) is None:
        cfg['grl_beta_cancer_target'] = 0.5
    if cfg.get('grl_beta_gamma', None) is None:
        cfg['grl_beta_gamma'] = 10.0
    if str(cfg.get('grl_beta_mode', 'dann')) not in {'dann', 'constant'}:
        raise ValueError(f"cfg['grl_beta_mode'] must be 'dann' or 'constant', got: {cfg.get('grl_beta_mode')}")

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

    # HPO / 复评入口（train_hpo.py 不执行常规训练）
    if args.tune is not None:
        if args.tune == 'all':
            return run_multi_stage_hpo(args, cfg, device)
        else:
            return run_hyperparameter_optimization(args, cfg, device)
    elif args.rescore_topk is not None:
        return run_rescore_multiple_stages(args, cfg, device)
    else:
        print("请使用 --tune 或 --rescore_topk 运行HPO/复评；常规训练请使用 train.py")
        return

if __name__ == '__main__':
    main()
