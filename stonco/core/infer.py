import argparse, os, numpy as np, torch
from stonco.utils.preprocessing import Preprocessor, GraphBuilder
from stonco.utils.utils import load_json, load_model_state_dict
from .models import STOnco_Classifier
from torch_geometric.data import Data as PyGData
import pandas as pd

# 新增：推理引擎类，封装模型/预处理/图构建与解释性计算
class InferenceEngine:
    def __init__(self, artifacts_dir: str, device: str | None = None, num_threads: int | None = None):
        self.artifacts_dir = artifacts_dir
        if num_threads is not None and num_threads > 0:
            try:
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(max(1, min(num_threads, 2)))
            except Exception as e:
                print(f"[Warn] set threads failed: {e}")
        meta = load_json(os.path.join(artifacts_dir, 'meta.json'))
        self.cfg = dict(meta.get('cfg', {}))
        # 兼容字段默认
        self.cfg.setdefault('lap_pe_dim', 16)
        self.cfg.setdefault('concat_lap_pe', True)
        self.cfg.setdefault('lap_pe_use_gaussian', False)
        self.cfg.setdefault('knn_k', 6)
        self.cfg.setdefault('gaussian_sigma_factor', 1.0)
        self.cfg.setdefault('model', 'gatv2')
        self.cfg.setdefault('heads', 4)
        self.cfg.setdefault('hidden', 128)
        self.cfg.setdefault('num_layers', 3)
        self.cfg.setdefault('dropout', 0.3)
        self.cfg.setdefault('clf_hidden', [256, 128, 64])

        self.pp = Preprocessor.load(artifacts_dir)
        self.device = torch.device(device) if isinstance(device, str) else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # 构建一个dummy图以确定输入维度
        # 注意：调用方通常会先transform并assemble，本处仅在需要时由外部传入in_dim
        self.model = None  # 延迟构建，待首个图到来时设置

    def build_model_if_needed(self, in_dim: int):
        if self.model is None:
            clf_hidden = self.cfg.get('clf_hidden', [256, 128, 64])
            if isinstance(clf_hidden, str):
                clf_hidden = [int(x.strip()) for x in clf_hidden.split(',') if x.strip() != '']
            clf_hidden = [int(x) for x in clf_hidden]
            m = STOnco_Classifier(
                in_dim=in_dim,
                hidden=self.cfg['hidden'],
                num_layers=self.cfg['num_layers'],
                dropout=self.cfg['dropout'],
                model=self.cfg['model'],
                heads=self.cfg.get('heads', 4),
                clf_hidden=clf_hidden,
            )
            _ = m.load_state_dict(load_model_state_dict(self.artifacts_dir, map_location=self.device), strict=False)
            self.model = m.to(self.device)
            self.model.eval()
        return self.model

    def assemble_pyg(self, Xp: np.ndarray, xy: np.ndarray) -> PyGData:
        gb = GraphBuilder(knn_k=self.cfg['knn_k'], gaussian_sigma_factor=self.cfg['gaussian_sigma_factor'])
        edge_index, edge_weight, _ = gb.build_knn(xy)
        if self.cfg.get('lap_pe_dim', 0) and self.cfg.get('lap_pe_dim', 0) > 0:
            pe = gb.lap_pe(edge_index, Xp.shape[0], k=self.cfg['lap_pe_dim'],
                           edge_weight=edge_weight if self.cfg.get('lap_pe_use_gaussian', False) else None,
                           use_gaussian_weights=self.cfg.get('lap_pe_use_gaussian', False))
        else:
            pe = None
        if self.cfg.get('concat_lap_pe', True) and pe is not None:
            x = np.hstack([Xp, pe]).astype('float32')
        else:
            x = Xp.astype('float32')
        data = PyGData(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index).long(), edge_weight=torch.from_numpy(edge_weight).float())
        data.num_nodes = x.shape[0]
        return data

    def predict_proba(self, g: PyGData) -> np.ndarray:
        g = g.to(self.device)
        self.build_model_if_needed(g.x.shape[1])
        with torch.no_grad():
            out = self.model(g.x, g.edge_index, batch=getattr(g, 'batch', None), edge_weight=getattr(g, 'edge_weight', None))
            probs = torch.sigmoid(out['logits']).detach().cpu().numpy()
        return probs

    def compute_gene_attr_for_graph(self, g: PyGData, method: str = 'ig', ig_steps: int = 50) -> np.ndarray:
        # 仅对基因特征部分求导（考虑PCA反投影）
        self.build_model_if_needed(g.x.shape[1])
        device = self.device
        pp = self.pp
        use_pca = bool(getattr(pp, 'use_pca', True) and getattr(pp, 'pca', None) is not None)
        feat_dim_gene = (int(pp.pca.n_components_) if use_pca else len(pp.hvg))
        gx = g.to(device)
        x = gx.x
        edge_index = gx.edge_index
        edge_weight = getattr(gx, 'edge_weight', None)
        x_in = x.detach().clone()
        baseline = x_in.detach().clone()
        baseline[:, :feat_dim_gene] = 0.0
        delta = (x_in - baseline)
        if method == 'ig':
            steps = max(1, int(ig_steps))
            grads_sum = torch.zeros_like(x_in[:, :feat_dim_gene])
            for s in range(1, steps + 1):
                alpha = float(s) / float(steps)
                x_step = baseline + delta * alpha
                x_step = x_step.detach().requires_grad_(True)
                out = self.model(x_step, edge_index, batch=getattr(gx, 'batch', None), edge_weight=edge_weight)
                logits = out['logits']
                loss = logits.sum()
                grad = torch.autograd.grad(loss, x_step, retain_graph=False)[0]
                grads_sum = grads_sum + grad[:, :feat_dim_gene].detach()
            avg_grads = grads_sum / float(steps)
            attr_feat = (delta[:, :feat_dim_gene] * avg_grads)
            attr_feat_graph = attr_feat.abs().mean(dim=0)
        else:
            x_sal = x_in.detach().clone().requires_grad_(True)
            out = self.model(x_sal, edge_index, batch=getattr(gx, 'batch', None), edge_weight=edge_weight)
            logits = out['logits']
            loss = logits.sum()
            grad = torch.autograd.grad(loss, x_sal, retain_graph=False)[0]
            sal = grad[:, :feat_dim_gene]
            attr_feat_graph = sal.abs().mean(dim=0)
        if use_pca:
            W = torch.from_numpy(pp.pca.components_.astype(np.float32)).to(attr_feat_graph.device)
            attr_z = attr_feat_graph @ W
        else:
            attr_z = attr_feat_graph
        scale = torch.from_numpy(pp.scaler.scale_.astype(np.float32)).to(attr_z.device)
        scale = torch.clamp(scale, min=1e-6)
        attr_gene = (attr_z / scale).detach().cpu().numpy()
        return attr_gene

def assemble_pyg(Xp, xy, cfg):
    from stonco.utils.preprocessing import GraphBuilder
    gb = GraphBuilder(knn_k=cfg['knn_k'], gaussian_sigma_factor=cfg['gaussian_sigma_factor'])
    edge_index, edge_weight, mean_nd = gb.build_knn(xy)
    # lapPE（可选：使用高斯权重并控制是否拼接）
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
    data = PyGData(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index).long(), edge_weight=torch.from_numpy(edge_weight).float())
    data.num_nodes = x.shape[0]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True, help='npz with X, xy, gene_names or dataset with Xs/xys')
    parser.add_argument('--artifacts_dir', default='artifacts')
    parser.add_argument('--out_csv', default='preds.csv')
    parser.add_argument('--index', type=int, default=None, help='Index of sample in NPZ if it contains Xs/xys; default 0')
    # 新增：CPU线程控制参数
    parser.add_argument('--num_threads', type=int, default=None, help='设置PyTorch计算线程数（CPU模式下限制核心占用)')
    # 新增：解释性输出参数（默认开启，可用 --no_explain 关闭）
    parser.add_argument('--explain_saliency', action='store_true', default=True, help='在推理时计算该样本的基因重要性向量并保存（默认开启）')
    parser.add_argument('--no_explain', action='store_false', dest='explain_saliency', help='关闭解释性输出')
    parser.add_argument('--explain_method', choices=['ig', 'saliency'], default='ig', help='选择解释方法')
    parser.add_argument('--ig_steps', type=int, default=50, help='IG步数，默认50')
    parser.add_argument('--gene_attr_out', default=None, help='保存该样本完整基因重要性向量的CSV路径（默认与out_csv同目录；若提供任意路径，将以其去扩展名+.csv保存）')
    args = parser.parse_args()

    # 应用CPU线程设置
    if args.num_threads is not None and args.num_threads > 0:
        try:
            torch.set_num_threads(args.num_threads)
            torch.set_num_interop_threads(max(1, min(args.num_threads, 2)))
            print(f"Set torch threads: intra-op={args.num_threads}, inter-op={max(1, min(args.num_threads, 2))}")
        except Exception as e:
            print(f"Warning: failed to set torch threads: {e}")

    # 使用推理引擎
    engine = InferenceEngine(args.artifacts_dir, num_threads=args.num_threads)
    cfg = engine.cfg

    data = np.load(args.npz, allow_pickle=True)
    files = set(data.files)
    if {'X', 'xy'}.issubset(files):
        X = data['X']
        xy = data['xy']
        sample_tag = 'single'
    elif {'Xs', 'xys'}.issubset(files):
        idx = 0 if args.index is None else args.index
        Xs = data['Xs']
        xys = data['xys']
        X = Xs[idx]
        xy = xys[idx]
        sample_tag = f'idx{idx}'
    else:
        raise ValueError(f"Unsupported NPZ structure. Found keys: {sorted(files)}")
    gene_names = list(data['gene_names'])

    Xp = engine.pp.transform(X, gene_names)
    data_g = engine.assemble_pyg(Xp, xy)

    probs = engine.predict_proba(data_g)

    df = pd.DataFrame({'spot_idx': np.arange(len(probs)), 'x': xy[:,0], 'y': xy[:,1], 'p_tumor': probs})
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print('Saved predictions to', args.out_csv)

    if args.explain_saliency:
        try:
            attr_gene = engine.compute_gene_attr_for_graph(data_g, method=args.explain_method, ig_steps=args.ig_steps)
            out_dir = os.path.dirname(args.out_csv) or '.'
            out_csv = (os.path.splitext(args.gene_attr_out)[0] + '.csv') if args.gene_attr_out else os.path.join(out_dir, f'per_slide_gene_saliency_{sample_tag}.csv')
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            pd.DataFrame({'gene': np.array(engine.pp.hvg), 'attr': attr_gene}).to_csv(out_csv, index=False)
            print('Saved gene importance CSV to', out_csv)
        except Exception as e:
            print('[Explain] Failed to compute/save gene vector:', e)

if __name__ == '__main__':
    main()
