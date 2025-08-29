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
