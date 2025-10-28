import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, GCNConv, LayerNorm, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter

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
        sum_w = scatter(e_w, dst, dim=0, dim_size=N, reduce='sum').clamp(min=1.0).view(-1, 1)
        out = out / sum_w
        # 线性变换并与根节点变换相加
        out = self.lin_neigh(out) + self.lin_root(x)
        return out

    def message(self, x_j, e_w):
        return e_w.view(-1, 1) * x_j

class GNNBackbone(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=3, dropout=0.3, model='gatv2', heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.model = model
        dim_prev = in_dim
        for i in range(num_layers):
            if model == 'gatv2':
                conv = GATv2Conv(dim_prev, hidden, heads=heads, concat=True, dropout=dropout)
                dim_next = hidden * heads
            elif model == 'gcn':
                conv = GCNConv(dim_prev, hidden, add_self_loops=True, normalize=True)
                dim_next = hidden
            else:  # 'sage'
                conv = WeightedSAGEConv(dim_prev, hidden)
                dim_next = hidden
            self.convs.append(conv)
            self.norms.append(LayerNorm(dim_next))
            dim_prev = dim_next
        self.out_dim = dim_prev
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            # 传递权重仅当卷积层支持时
            if isinstance(conv, (WeightedSAGEConv, GCNConv)) and edge_weight is not None:
                h = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = conv(h, edge_index)
            h = F.relu(h)
            h = norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class ClassifierHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, h):
        return self.fc(h).squeeze(-1)

class DomainHead(nn.Module):
    def __init__(self, in_dim, n_domains, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains)
        )
    def forward(self, h):
        return self.net(h)

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
    def __init__(self, in_dim, hidden=128, num_layers=3, dropout=0.3, model='gatv2', heads=4,
                 use_domain_adv=False, n_domains=None, domain_hidden=64,
                 # 新增双域控制
                 use_domain_adv_slide=False, n_domains_slide=None,
                 use_domain_adv_cancer=False, n_domains_cancer=None):
        super().__init__()
        self.gnn = GNNBackbone(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=dropout, model=model, heads=heads)
        self.clf = ClassifierHead(self.gnn.out_dim)
        # 旧字段兼容
        self.use_domain_adv = use_domain_adv
        self.dom = None  # 保留旧属性以兼容
        # 新域对抗开关与head
        self.use_domain_adv_slide = use_domain_adv_slide or (use_domain_adv and (n_domains is not None))
        self.use_domain_adv_cancer = use_domain_adv_cancer
        self.dom_slide = DomainHead(self.gnn.out_dim, n_domains_slide if n_domains_slide is not None else (n_domains if n_domains is not None else 0), hidden=domain_hidden) \
            if self.use_domain_adv_slide and ((n_domains_slide is not None and n_domains_slide>0) or (n_domains is not None and n_domains>0)) else None
        self.dom_cancer = DomainHead(self.gnn.out_dim, n_domains_cancer, hidden=domain_hidden) \
            if self.use_domain_adv_cancer and (n_domains_cancer is not None and n_domains_cancer>0) else None

    def forward(self, x, edge_index, batch=None, edge_weight=None, domain_lambda=1.0, lambda_slide=1.0, lambda_cancer=1.0):
        h = self.gnn(x, edge_index, edge_weight=edge_weight)
        logits = self.clf(h)
        dom_logits_slide = None
        dom_logits_cancer = None
        if (self.dom_slide is not None) or (self.dom_cancer is not None):
            # 默认采用全图池化，如需图批次请传 batch
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            pooled = global_mean_pool(h, batch)
            if self.dom_slide is not None:
                dom_logits_slide = self.dom_slide(grad_reverse(pooled, lambda_slide))
            if self.dom_cancer is not None:
                dom_logits_cancer = self.dom_cancer(grad_reverse(pooled, lambda_cancer))
        # 保持旧键兼容：dom_logits 指向 slide 的输出
        return {'logits': logits, 'h': h, 'dom_logits_slide': dom_logits_slide, 'dom_logits_cancer': dom_logits_cancer, 'dom_logits': dom_logits_slide}


# Backward compatibility alias
STRIDE_Classifier = STOnco_Classifier
