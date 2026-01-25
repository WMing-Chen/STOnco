import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, global_mean_pool


class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转层（GRL）：前向恒等，反向乘以 -lambda。
    常用于域对抗训练。
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    return GradientReversalFunction.apply(x, lambd)


class GATv2Backbone(nn.Module):
    """
    仅包含 GATv2 卷积的GNN骨干网络。
    - num_layers：堆叠层数
    - hidden：每层隐藏维度
    - heads：多头注意力头数
    - dropout：层间dropout
    输出属性：out_dim（供分类器/域分类器使用）
    """

    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 3, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim_prev = in_dim
        for _ in range(num_layers):
            conv = GATv2Conv(dim_prev, hidden, heads=heads, concat=True, dropout=dropout)
            dim_next = hidden * heads
            self.convs.append(conv)
            self.norms.append(LayerNorm(dim_next))
            dim_prev = dim_next
        self.out_dim = dim_prev
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)


class DomainHead(nn.Module):
    def __init__(self, in_dim: int, n_domains: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains)
        )

    def forward(self, h):
        return self.net(h)


class SpotoncoGATv2Classifier(nn.Module):
    """
    仅基于 GATv2 的图分类模型，支持可选的双域对抗头（切片域与癌种域）。
    forward 返回：
      {
        'logits': 节点级logits,
        'h': 节点特征,
        'dom_logits_slide': 切片域logits或None,
        'dom_logits_cancer': 癌种域logits或None,
      }
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        # 域对抗相关
        use_domain_adv_slide: bool = False,
        n_domains_slide: int | None = None,
        use_domain_adv_cancer: bool = False,
        n_domains_cancer: int | None = None,
        domain_hidden: int = 64,
    ):
        super().__init__()
        self.gnn = GATv2Backbone(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=dropout, heads=heads)
        self.clf = ClassifierHead(self.gnn.out_dim)

        # 切片域头
        self.use_domain_adv_slide = use_domain_adv_slide and (n_domains_slide is not None and n_domains_slide > 0)
        self.dom_slide = DomainHead(self.gnn.out_dim, n_domains_slide, hidden=domain_hidden) if self.use_domain_adv_slide else None

        # 癌种域头
        self.use_domain_adv_cancer = use_domain_adv_cancer and (n_domains_cancer is not None and n_domains_cancer > 0)
        self.dom_cancer = DomainHead(self.gnn.out_dim, n_domains_cancer, hidden=domain_hidden) if self.use_domain_adv_cancer else None

    def forward(
        self,
        x,
        edge_index,
        batch=None,
        lambda_slide: float = 1.0,
        lambda_cancer: float = 1.0,
    ):
        h = self.gnn(x, edge_index)
        logits = self.clf(h)

        dom_logits_slide = None
        dom_logits_cancer = None

        if (self.dom_slide is not None) or (self.dom_cancer is not None):
            # 若未提供batch，默认整图为一个batch
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            pooled = global_mean_pool(h, batch)
            if self.dom_slide is not None:
                dom_logits_slide = self.dom_slide(grad_reverse(pooled, lambda_slide))
            if self.dom_cancer is not None:
                dom_logits_cancer = self.dom_cancer(grad_reverse(pooled, lambda_cancer))

        return {
            'logits': logits,
            'h': h,
            'dom_logits_slide': dom_logits_slide,
            'dom_logits_cancer': dom_logits_cancer,
        }


__all__ = [
    'GATv2Backbone',
    'ClassifierHead',
    'DomainHead',
    'SpotoncoGATv2Classifier',
]
