import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, GCNConv, LayerNorm
from torch_geometric.nn.conv import MessagePassing
from stonco.utils.utils import normalize_gnn_hidden


def _scatter_sum_1d(src, index, dim_size):
    out = src.new_zeros(int(dim_size))
    return out.scatter_add_(0, index, src)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradientReversalFunction.apply(x, lambd)

class WeightedSAGEConv(MessagePassing):
    """
    支持边权的GraphSAGE（Mean）卷积：对邻居进行加权平均，随后与中心节点特征线性变换相加。
    message: m_ij = w_ij * x_j
    aggregate: sum_j m_ij，然后除以 sum_j w_ij（若无权则w_ij=1，即普通平均）
    update: lin_neigh(agg) + lin_root(x_i)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, edge_weight=None):
        N = x.size(0)
        if edge_weight is None:
            e_w = x.new_ones(edge_index.size(1))
        else:
            e_w = edge_weight
        # 先聚合原始特征（未线性变换），得到加权和
        out = self.propagate(edge_index, x=x, e_w=e_w, size=(N, N))
        # 归一化：除以每个节点的权重和，若无权则为度数
        dst = edge_index[1]
        sum_w = _scatter_sum_1d(e_w, dst, N).clamp(min=1.0).view(-1, 1)
        out = out / sum_w
        # 线性变换并与根节点变换相加
        out = self.lin_neigh(out) + self.lin_root(x)
        return out

    def message(self, x_j, e_w):
        return e_w.view(-1, 1) * x_j

class GNNBackbone(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), num_layers=3, dropout=0.3, model='gatv2', heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.model = model
        dim_prev = in_dim
        hidden_list, effective_num_layers = normalize_gnn_hidden(hidden=hidden, num_layers=num_layers)
        self.hidden_dims = list(hidden_list)
        self.num_layers = int(effective_num_layers)
        for hidden_dim in self.hidden_dims:
            if model == 'gatv2':
                conv = GATv2Conv(dim_prev, hidden_dim, heads=heads, concat=True, dropout=dropout)
                dim_next = hidden_dim * heads
            elif model == 'gcn':
                conv = GCNConv(dim_prev, hidden_dim, add_self_loops=True, normalize=True)
                dim_next = hidden_dim
            else:  # 'sage'
                conv = WeightedSAGEConv(dim_prev, hidden_dim)
                dim_next = hidden_dim
            self.convs.append(conv)
            self.norms.append(LayerNorm(dim_next))
            dim_prev = dim_next
        self.out_dim = dim_prev
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, node_mask=None):
        h = x
        if node_mask is not None:
            h = h * node_mask.to(device=h.device, dtype=h.dtype).view(-1, 1)
        for conv, norm in zip(self.convs, self.norms):
            # 传递权重仅当卷积层支持时
            if isinstance(conv, (WeightedSAGEConv, GCNConv)) and edge_weight is not None:
                h = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = conv(h, edge_index)
            h = F.relu(h)
            h = norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if node_mask is not None:
                h = h * node_mask.to(device=h.device, dtype=h.dtype).view(-1, 1)
        return h

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (256, 128, 64)
        hidden_dims = tuple(int(x) for x in hidden_dims)
        if len(hidden_dims) < 1:
            raise ValueError(f'clf_hidden must contain at least 1 integer, got {hidden_dims}')
        if any(int(h) <= 0 for h in hidden_dims):
            raise ValueError(f'clf_hidden must contain only positive integers, got {hidden_dims}')

        self.hidden_dims = hidden_dims
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        dim_prev = int(in_dim)
        for hidden_dim in self.hidden_dims:
            hidden_dim = int(hidden_dim)
            self.hidden_layers.append(nn.Linear(dim_prev, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))
            dim_prev = hidden_dim
        self.fc_out = nn.Linear(dim_prev, 1)
        self.drop = nn.Dropout(float(dropout))
        self.latent_dim = int(self.hidden_dims[-1])

    def forward(self, h, return_z=False):
        x = h
        for fc, bn in zip(self.hidden_layers, self.hidden_bns):
            x = fc(x)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)
        z_clf = x
        logits = self.fc_out(z_clf).squeeze(-1)
        if return_z:
            return logits, z_clf
        return logits, None

class DomainHead(nn.Module):
    def __init__(self, in_dim, n_domains, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, n_domains)
        )
    def forward(self, h):
        return self.net(h)


class ResidualGatedFusion(nn.Module):
    def __init__(self, gene_dim, img_dim, gate_hidden=128, dropout=0.0):
        super().__init__()
        self.gene_dim = int(gene_dim)
        self.img_dim = int(img_dim)
        self.img_proj = nn.Linear(self.img_dim, self.gene_dim)
        self.ln_gene = nn.LayerNorm(self.gene_dim)
        self.ln_img = nn.LayerNorm(self.gene_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.gene_dim * 2 + 1, int(gate_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(gate_hidden), 1),
        )

    def forward(self, h_gene, h_img, img_mask):
        if img_mask is None:
            raise ValueError('img_mask is required for residual gated fusion')
        mask = img_mask.to(device=h_gene.device, dtype=h_gene.dtype).view(-1, 1)
        u_img = self.img_proj(h_img) * mask
        gate_input = torch.cat([self.ln_gene(h_gene), self.ln_img(u_img), mask], dim=1)
        gate = torch.sigmoid(self.gate_mlp(gate_input)) * mask
        img_residual = gate * u_img
        h_fused = h_gene + img_residual
        stats = {
            'gate': gate,
            'img_residual': img_residual,
            'u_img': u_img,
        }
        return h_fused, stats

class STOnco_Classifier(nn.Module):
    """
    Unified STOnco Classifier with optional dual-domain adversarial learning.
    
    This is a merged model that contains:
    - GNNBackbone: The main GNN encoder
    - ClassifierHead: For tumor/non-tumor classification
    - DomainHead (optional): For dual-domain adversarial learning
    
    Args:
        backbone_config: Configuration for the GNN backbone
        classifier_config: Configuration for the classifier head
        domain_config: Configuration for domain heads (optional)
        use_domain_adaptation: Whether to enable domain adaptation
    """
    def __init__(self, in_dim, hidden=(256, 128, 64), num_layers=3, dropout=0.3, model='gatv2', heads=4,
                 gnn_dropout=None,
                 clf_dropout=None,
                 dom_dropout=None,
                 clf_hidden=(256, 128, 64),
                 domain_hidden=64,
                 use_domain_adv_slide=False, n_domains_slide=None,
                 use_domain_adv_cancer=False, n_domains_cancer=None,
                 image_fusion_mode='early_concat',
                 in_dim_img=None,
                 img_hidden=(128, 64),
                 img_num_layers=2,
                 img_model=None,
                 img_heads=None,
                 img_dropout=None,
                 fusion_gate_hidden=128,
                 fusion_dropout=0.0):
        super().__init__()
        if gnn_dropout is None:
            gnn_dropout = dropout
        if clf_dropout is None:
            clf_dropout = dropout
        if dom_dropout is None:
            dom_dropout = dropout
        self.gnn_dropout = float(gnn_dropout)
        self.clf_dropout = float(clf_dropout)
        self.dom_dropout = float(dom_dropout)
        self.image_fusion_mode = str(image_fusion_mode or 'early_concat').lower()
        if self.image_fusion_mode not in {'early_concat', 'dual_branch_residual_gate'}:
            raise ValueError(
                "image_fusion_mode must be one of 'early_concat' or 'dual_branch_residual_gate', "
                f'got: {image_fusion_mode}'
            )

        if self.image_fusion_mode == 'dual_branch_residual_gate':
            if in_dim_img is None:
                raise ValueError("in_dim_img is required when image_fusion_mode='dual_branch_residual_gate'")
            if img_dropout is None:
                img_dropout = self.gnn_dropout
            if img_model is None:
                img_model = model
            if img_heads is None:
                img_heads = heads
            self.gnn_gene = GNNBackbone(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=self.gnn_dropout, model=model, heads=heads)
            self.gnn_img = GNNBackbone(
                in_dim=int(in_dim_img),
                hidden=img_hidden,
                num_layers=img_num_layers,
                dropout=float(img_dropout),
                model=img_model,
                heads=int(img_heads),
            )
            self.fusion = ResidualGatedFusion(
                gene_dim=self.gnn_gene.out_dim,
                img_dim=self.gnn_img.out_dim,
                gate_hidden=int(fusion_gate_hidden),
                dropout=float(fusion_dropout),
            )
            self.encoder_out_dim = int(self.gnn_gene.out_dim)
            self.gnn = self.gnn_gene
        else:
            self.gnn = GNNBackbone(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=self.gnn_dropout, model=model, heads=heads)
            self.gnn_gene = None
            self.gnn_img = None
            self.fusion = None
            self.encoder_out_dim = int(self.gnn.out_dim)

        self.clf = ClassifierHead(self.encoder_out_dim, hidden_dims=clf_hidden, dropout=self.clf_dropout)
        self.use_domain_adv_slide = bool(use_domain_adv_slide) and (n_domains_slide is not None and int(n_domains_slide) > 0)
        self.use_domain_adv_cancer = bool(use_domain_adv_cancer) and (n_domains_cancer is not None and int(n_domains_cancer) > 0)
        self.dom_slide = DomainHead(self.encoder_out_dim, int(n_domains_slide), hidden=domain_hidden, dropout=self.dom_dropout) if self.use_domain_adv_slide else None
        self.dom_cancer = DomainHead(self.encoder_out_dim, n_domains_cancer, hidden=domain_hidden, dropout=self.dom_dropout) \
            if self.use_domain_adv_cancer else None

    def encode(self, x=None, edge_index=None, edge_weight=None, x_gene=None, x_img=None, img_mask=None, pe_gene=None):
        if edge_index is None:
            raise ValueError('edge_index is required')
        if self.image_fusion_mode == 'early_concat':
            if x is None:
                x = x_gene
            if x is None:
                raise ValueError("x is required when image_fusion_mode='early_concat'")
            h = self.gnn(x, edge_index, edge_weight=edge_weight)
            return h

        if x_gene is None:
            x_gene = x
        if x_gene is None:
            raise ValueError("x_gene is required when image_fusion_mode='dual_branch_residual_gate'")
        if x_img is None:
            raise ValueError("x_img is required when image_fusion_mode='dual_branch_residual_gate'")
        if img_mask is None:
            raise ValueError("img_mask is required when image_fusion_mode='dual_branch_residual_gate'")
        if pe_gene is not None:
            x_gene = torch.cat([x_gene, pe_gene], dim=1)
        mask = img_mask.to(device=x_img.device, dtype=x_img.dtype).view(-1, 1)
        h_gene = self.gnn_gene(x_gene, edge_index, edge_weight=edge_weight)
        h_img = self.gnn_img(x_img * mask, edge_index, edge_weight=edge_weight, node_mask=img_mask)
        h_fused, fusion_stats = self.fusion(h_gene, h_img, img_mask)
        self._last_fusion_stats = fusion_stats
        return h_fused

    def forward(
        self,
        x,
        edge_index,
        batch=None,
        edge_weight=None,
        grl_beta_slide=1.0,
        grl_beta_cancer=1.0,
        return_z=False,
        return_h=False,
        x_gene=None,
        x_img=None,
        img_mask=None,
        pe_gene=None,
        return_fusion_stats=False,
    ):
        h = self.encode(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            x_gene=x_gene,
            x_img=x_img,
            img_mask=img_mask,
            pe_gene=pe_gene,
        )
        logits, z_clf = self.clf(h, return_z=return_z)
        dom_logits_slide = None
        dom_logits_cancer = None
        if (self.dom_slide is not None) or (self.dom_cancer is not None):
            if self.dom_slide is not None:
                dom_logits_slide = self.dom_slide(grad_reverse(h, grl_beta_slide))
            if self.dom_cancer is not None:
                dom_logits_cancer = self.dom_cancer(grad_reverse(h, grl_beta_cancer))
        out = {'logits': logits, 'dom_logits_slide': dom_logits_slide, 'dom_logits_cancer': dom_logits_cancer}
        if return_z:
            out['z_clf'] = z_clf
            if int(z_clf.shape[1]) == 64:
                out['z64'] = z_clf
        if return_h:
            out['h'] = h
        if return_fusion_stats and self.image_fusion_mode == 'dual_branch_residual_gate':
            stats = getattr(self, '_last_fusion_stats', None)
            if stats is not None:
                out['fusion_stats'] = stats
        return out
