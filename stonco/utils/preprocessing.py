import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh
import math, os, joblib
try:
    import scanpy as sc
    _SCANPY = True
except Exception:
    _SCANPY = False

def build_node_features_early_fusion(Xp_gene, cfg, Xp_img=None, img_mask=None, pe=None):
    xs = [Xp_gene]
    if Xp_img is not None:
        if img_mask is None:
            raise ValueError('img_mask is required when Xp_img is provided')
        img_mask = np.asarray(img_mask).reshape(-1)
        if img_mask.shape[0] != Xp_gene.shape[0]:
            raise ValueError(f'img_mask length mismatch: {img_mask.shape[0]} vs n_spots={Xp_gene.shape[0]}')
        img_mask_col = img_mask.astype('float32').reshape(-1, 1)
        xs.extend([Xp_img, img_mask_col])

    x = np.hstack(xs).astype('float32')
    if cfg.get('concat_lap_pe', True) and pe is not None:
        x = np.hstack([x, pe]).astype('float32')
    return x


class Preprocessor:
    def __init__(self, n_hvg=2000, norm_target=1e4, do_log1p=True, pca_dim=64, zclip=5.0, use_pca=True):
        self.n_hvg = n_hvg
        self.norm_target = norm_target
        self.do_log1p = do_log1p
        self.pca_dim = pca_dim
        self.zclip = zclip
        self.use_pca = use_pca
        self.hvg = None
        self.scaler = StandardScaler()
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_dim)
        else:
            self.pca = None
        self.fitted = False

    def cp10k_log1p(self, X):
        lib = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
        Xn = X / lib * self.norm_target
        if self.do_log1p:
            Xn = np.log1p(Xn)
        return Xn

    def select_hvg(self, X_list, gene_names):
        if _SCANPY and len(X_list) > 0:
            import anndata as ad
            X_cat = np.vstack(X_list)
            adata = ad.AnnData(X_cat)
            adata.var_names = gene_names
            sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor='seurat_v3')
            self.hvg = list(adata.var_names[adata.var['highly_variable']])
        else:
            X_cat = np.vstack(X_list)
            vars_ = X_cat.var(axis=0)
            idx = np.argsort(vars_)[::-1][:self.n_hvg]
            self.hvg = [gene_names[i] for i in idx]

    def fit(self, slides, gene_names):
        Xn_list = [self.cp10k_log1p(s['X']) for s in slides]
        if self.hvg is None:
            self.select_hvg(Xn_list, gene_names)
        idx = [gene_names.index(g) for g in self.hvg]
        Xh = np.vstack([Xn[:, idx] for Xn in Xn_list])
        # clip extremes then fit scaler and pca
        low, high = np.percentile(Xh, 1), np.percentile(Xh, 99)
        Xh = np.clip(Xh, low, high)
        self.scaler.fit(Xh)
        Xz = self.scaler.transform(Xh)
        Xz = np.clip(Xz, -self.zclip, self.zclip)
        if self.use_pca:
            self.pca.fit(Xz)
        self.fitted = True

    def transform(self, X, gene_names):
        assert self.fitted, 'Preprocessor not fitted'
        Xn = self.cp10k_log1p(X)
        idx = []
        name_to_idx = {g:i for i,g in enumerate(gene_names)}
        for g in self.hvg:
            idx.append(name_to_idx.get(g, -1))
        idx = np.array(idx)
        Xh = np.zeros((Xn.shape[0], len(self.hvg)), dtype=Xn.dtype)
        present = idx >= 0
        if present.any():
            Xh[:, present] = Xn[:, idx[present]]
        low, high = np.percentile(Xh, 1), np.percentile(Xh, 99)
        Xh = np.clip(Xh, low, high)
        Xz = self.scaler.transform(Xh)
        Xz = np.clip(Xz, -self.zclip, self.zclip)
        if self.use_pca:
            Xp = self.pca.transform(Xz)
            return Xp
        else:
            # 直接返回Z-score后的特征
            return Xz

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'genes_hvg.txt'), 'w') as f:
            for g in self.hvg:
                f.write(g + '\n')
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        # 记录是否使用PCA
        joblib.dump({'use_pca': self.use_pca, 'pca': self.pca}, os.path.join(path, 'pca.joblib'))

    @classmethod
    def load(cls, path):
        # 尝试向后兼容旧格式：旧格式的pca.joblib直接是PCA对象
        with open(os.path.join(path, 'genes_hvg.txt'), 'r') as f:
            hvg = [line.strip() for line in f]
        scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
        pca_obj = joblib.load(os.path.join(path, 'pca.joblib'))
        if isinstance(pca_obj, dict):
            use_pca = bool(pca_obj.get('use_pca', True))
            pca = pca_obj.get('pca', None)
        else:
            use_pca = True
            pca = pca_obj
        inst = cls(use_pca=use_pca)
        inst.hvg = hvg
        inst.scaler = scaler
        inst.pca = pca
        inst.fitted = True
        return inst

class ImagePreprocessor:
    def __init__(self, img_use_pca=True, img_pca_dim=256):
        self.use_pca = bool(img_use_pca)
        self.pca_dim = int(img_pca_dim)
        self.scaler = StandardScaler()
        if self.use_pca:
            self.pca = PCA(
                n_components=self.pca_dim,
                whiten=False,
                svd_solver='randomized',
                random_state=42,
            )
        else:
            self.pca = None
        self.feature_names = None
        self.fitted = False

    def fit(self, X_img_list, img_masks_list=None):
        X_list = []
        if img_masks_list is None:
            for X_img in X_img_list:
                X_list.append(np.asarray(X_img, dtype=np.float32))
        else:
            for X_img, m in zip(X_img_list, img_masks_list):
                X_img = np.asarray(X_img, dtype=np.float32)
                m = np.asarray(m).reshape(-1)
                if m.shape[0] != X_img.shape[0]:
                    raise ValueError(f'img_mask length mismatch: {m.shape[0]} vs X_img rows={X_img.shape[0]}')
                idx = (m.astype(np.uint8) == 1)
                if idx.any():
                    X_list.append(X_img[idx])
        if not X_list:
            raise ValueError('No valid image features to fit ImagePreprocessor (all img_mask==0).')
        X_fit = np.vstack(X_list)
        if self.use_pca and X_fit.shape[0] < self.pca_dim:
            raise ValueError(
                f'img_use_pca=1 requires n_valid_spots({X_fit.shape[0]}) >= img_pca_dim({self.pca_dim}).'
            )
        self.scaler.fit(X_fit)
        Xz = self.scaler.transform(X_fit)
        if self.use_pca:
            self.pca.fit(Xz)
        self.fitted = True

    def out_dim(self):
        if self.use_pca and self.pca is not None:
            return int(getattr(self.pca, 'n_components_', self.pca_dim))
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            return int(self.scaler.mean_.shape[0])
        return int(self.pca_dim) if self.use_pca else 2048

    def transform(self, X_img, img_mask=None):
        assert self.fitted, 'ImagePreprocessor not fitted'
        X_img = np.asarray(X_img, dtype=np.float32)
        if X_img.ndim != 2:
            raise ValueError(f'X_img must be 2D, got shape={X_img.shape}')
        n = int(X_img.shape[0])
        out_dim = int(self.out_dim())
        Xp = np.zeros((n, out_dim), dtype=np.float32)
        if img_mask is None:
            idx = np.arange(n, dtype=int)
        else:
            img_mask = np.asarray(img_mask).reshape(-1)
            if img_mask.shape[0] != n:
                raise ValueError(f'img_mask length mismatch: {img_mask.shape[0]} vs X_img rows={n}')
            idx = np.where(img_mask.astype(np.uint8) == 1)[0]
        if idx.size == 0:
            return Xp

        X_valid = X_img[idx]
        Xz = self.scaler.transform(X_valid)
        if self.use_pca and self.pca is not None:
            Xz = self.pca.transform(Xz)
        Xp[idx] = Xz.astype(np.float32)
        return Xp

    def save(self, artifacts_dir, img_feature_names=None):
        assert self.fitted, 'ImagePreprocessor not fitted'
        os.makedirs(artifacts_dir, exist_ok=True)
        if img_feature_names is not None:
            with open(os.path.join(artifacts_dir, 'img_feature_names.txt'), 'w') as f:
                for name in img_feature_names:
                    f.write(str(name) + '\n')
            self.feature_names = list(map(str, img_feature_names))
        joblib.dump(self.scaler, os.path.join(artifacts_dir, 'img_scaler.joblib'))
        if self.use_pca and self.pca is not None:
            joblib.dump(self.pca, os.path.join(artifacts_dir, 'img_pca.joblib'))

    @classmethod
    def load(cls, artifacts_dir, img_use_pca=True):
        scaler_path = os.path.join(artifacts_dir, 'img_scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f'img_scaler.joblib not found in artifacts_dir: {artifacts_dir}')
        scaler = joblib.load(scaler_path)
        inst = cls(img_use_pca=bool(img_use_pca), img_pca_dim=256)
        inst.scaler = scaler

        pca_path = os.path.join(artifacts_dir, 'img_pca.joblib')
        if bool(img_use_pca):
            if not os.path.exists(pca_path):
                raise FileNotFoundError(f'img_pca.joblib not found in artifacts_dir: {artifacts_dir}')
            inst.pca = joblib.load(pca_path)
            inst.pca_dim = int(getattr(inst.pca, 'n_components_', inst.pca_dim))
            inst.use_pca = True
        else:
            inst.pca = None
            inst.use_pca = False

        names_path = os.path.join(artifacts_dir, 'img_feature_names.txt')
        if os.path.exists(names_path):
            with open(names_path, 'r') as f:
                inst.feature_names = [line.strip() for line in f if line.strip() != '']
        inst.fitted = True
        return inst

class GraphBuilder:
    def __init__(self, knn_k=6, gaussian_sigma_factor=1.0):
        self.knn_k = knn_k
        self.gaussian_sigma_factor = gaussian_sigma_factor

    def build_knn(self, xy):
        nn = NearestNeighbors(n_neighbors=self.knn_k + 1, algorithm='kd_tree')
        nn.fit(xy)
        dists, idxs = nn.kneighbors(xy, return_distance=True)
        mean_nd = float(dists[:,1:].mean())
        sigma = max(1e-6, self.gaussian_sigma_factor * mean_nd)
        sigma2 = sigma*sigma
        src_list, dst_list, w_list = [], [], []
        n = xy.shape[0]
        for i in range(n):
            for j, d in zip(idxs[i,1:], dists[i,1:]):
                src_list.append(i); dst_list.append(int(j)); w_list.append(math.exp(- (d*d)/(2*sigma2)))
        src = src_list + dst_list
        dst = dst_list + src_list
        w = w_list + w_list
        import numpy as np
        edge_index = np.vstack([np.array(src), np.array(dst)])
        return edge_index, np.array(w, dtype=float), mean_nd

    @staticmethod
    def lap_pe(edge_index, num_nodes, k=16, edge_weight=None, use_gaussian_weights=False):
        """
        计算拉普拉斯位置编码。
        
        Args:
            edge_index: 边索引 (2, num_edges)
            num_nodes: 节点数量
            k: 拉普拉斯特征向量的维度
            edge_weight: 边权重 (num_edges,)，仅在 use_gaussian_weights=True 时使用
            use_gaussian_weights: 是否使用高斯距离权重计算加权拉普拉斯
        
        Returns:
            PE: 拉普拉斯位置编码 (num_nodes, k)
        """
        import numpy as np
        src, dst = edge_index
        
        if use_gaussian_weights and edge_weight is not None:
            # 使用加权拉普拉斯
            data = edge_weight.astype(float)
        else:
            # 使用无权拉普拉斯
            data = np.ones(src.shape[0], dtype=float)
            
        A = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
        A = (A + A.T) * 0.5
        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
        L = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
        k_eff = min(k + 1, num_nodes - 1)
        try:
            vals, vecs = eigsh(L, k=k_eff, which='SM')
            if vecs.shape[1] > 1:
                PE = vecs[:, 1:k_eff]
            else:
                PE = np.zeros((num_nodes, 0))
        except Exception:
            PE = np.zeros((num_nodes, k))
        if PE.shape[1] < k:
            pad = np.zeros((num_nodes, k - PE.shape[1]))
            PE = np.hstack([PE, pad])
        return PE.astype('float32')
