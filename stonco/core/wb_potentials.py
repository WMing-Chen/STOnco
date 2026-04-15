import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_potential_mlp(in_dim, hidden):
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden)),
        nn.LeakyReLU(0.2),
        nn.Linear(int(hidden), int(hidden)),
        nn.LeakyReLU(0.2),
        nn.Linear(int(hidden), 1),
    )


class GeneratedSupportMap(nn.Module):
    """Identity-initialized residual map: b = h + MLP(LayerNorm(h))."""

    def __init__(self, dim, hidden=128, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(dim)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return h + self.net(self.norm(h))


class SinglePotentialBank(nn.Module):
    """Scalar potential bank for euclidean_pairwise generated-support loss."""

    def __init__(self, n_domains, in_dim, hidden=128):
        super().__init__()
        self.n_domains = int(n_domains)
        self.heads = nn.ModuleList([
            _make_potential_mlp(in_dim, hidden)
            for _ in range(self.n_domains)
        ])

    def forward_one_domain(self, b, domain_idx):
        return self.heads[int(domain_idx)](b).squeeze(-1)

    def forward_by_domain(self, b, domain_idx):
        if b.numel() == 0:
            return b.new_empty((0,))
        out = b.new_empty(b.size(0))
        for k in torch.unique(domain_idx.detach()).tolist():
            mask = domain_idx == int(k)
            out[mask] = self.forward_one_domain(b[mask], int(k))
        return out


class DualPotentialBank(nn.Module):
    """Source-side f_k and support-side g_k potentials for dual WB loss."""

    def __init__(self, n_domains, in_dim, hidden=128):
        super().__init__()
        self.n_domains = int(n_domains)
        self.f = nn.ModuleList([
            _make_potential_mlp(in_dim, hidden)
            for _ in range(self.n_domains)
        ])
        self.g = nn.ModuleList([
            _make_potential_mlp(in_dim, hidden)
            for _ in range(self.n_domains)
        ])

    def f_one_domain(self, h, domain_idx):
        return self.f[int(domain_idx)](h).squeeze(-1)

    def g_centered(self, b_support, active_domains, weights=None):
        active_domains = [int(k) for k in active_domains]
        if len(active_domains) == 0:
            return b_support.new_empty((0, b_support.size(0)))
        if weights is None:
            weights = b_support.new_full((len(active_domains),), 1.0 / len(active_domains))
        else:
            weights = weights.to(device=b_support.device, dtype=b_support.dtype)
        all_g = torch.stack([self.g[k](b_support).squeeze(-1) for k in active_domains], dim=0)
        g_mean = (weights.view(-1, 1) * all_g).sum(dim=0)
        return all_g - g_mean.unsqueeze(0)


class GeneratedSupportWBLoss(nn.Module):
    """Generated-support Wasserstein barycenter losses used by train.py.

    model_loss returns a raw WB alignment term and a separate anchor loss.
    The caller decides how to weight the anchor term in the total loss.
    """

    def __init__(
        self,
        n_domains,
        in_dim,
        loss_type='euclidean_pairwise',
        potential_hidden=128,
        spots_per_graph=64,
        spots_per_cancer=0,
        support_size=128,
        min_cancers=2,
        min_spots=2,
        regularizer='l2',
        epsilon=0.1,
        label_balanced_sampling=False,
        state_direction=False,
        state_direction_weight=0.1,
        euclid_pairwise_weight=1.0,
        potential_weight=1.0,
        potential_constraint_weight=0.01,
    ):
        super().__init__()
        self.n_domains = int(n_domains)
        self.in_dim = int(in_dim)
        self.loss_type = str(loss_type).lower()
        self.spots_per_graph = int(spots_per_graph)
        self.spots_per_cancer = int(spots_per_cancer)
        self.support_size = int(support_size)
        self.min_cancers = int(min_cancers)
        self.min_spots = int(min_spots)
        self.regularizer = str(regularizer).lower()
        self.epsilon = float(epsilon)
        self.label_balanced_sampling = bool(label_balanced_sampling)
        self.state_direction_weight = float(state_direction_weight)
        self.euclid_pairwise_weight = float(euclid_pairwise_weight)
        self.potential_weight = float(potential_weight)
        self.potential_constraint_weight = float(potential_constraint_weight)

        if self.loss_type == 'euclidean_pairwise':
            self.potentials = SinglePotentialBank(self.n_domains, self.in_dim, hidden=potential_hidden)
        elif self.loss_type == 'dual_potential':
            self.potentials = DualPotentialBank(self.n_domains, self.in_dim, hidden=potential_hidden)
        else:
            raise ValueError(f"wb_loss_type must be dual_potential or euclidean_pairwise, got: {loss_type}")

        if self.regularizer not in {'l2', 'entropy'}:
            raise ValueError(f"wb_regularizer must be l2 or entropy, got: {regularizer}")

        self.state_direction = None
        if bool(state_direction):
            v = torch.empty(self.in_dim)
            nn.init.normal_(v, mean=0.0, std=1.0 / math.sqrt(max(self.in_dim, 1)))
            self.state_direction = nn.Parameter(v)

    def potential_parameters(self):
        return self.potentials.parameters()

    def main_parameters(self):
        if self.state_direction is None:
            return []
        return [self.state_direction]

    def _zero(self, ref, requires_grad=False):
        z = ref.new_tensor(0.0)
        if requires_grad:
            z = z + ref.sum() * 0.0
        return z

    def _normalize_latent(self, x):
        return F.layer_norm(x, (x.size(-1),))

    def _select_indices(self, cancer_dom, graph_nodes=None, y=None):
        device = cancer_dom.device
        if cancer_dom.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if graph_nodes is None or int(self.spots_per_graph) <= 0:
            idx = torch.arange(cancer_dom.numel(), device=device)
        else:
            keep = []
            for gid in sorted(int(v) for v in torch.unique(graph_nodes.detach()).tolist()):
                idx_g = torch.where(graph_nodes == gid)[0]
                if idx_g.numel() <= int(self.spots_per_graph):
                    keep.append(idx_g)
                else:
                    perm = torch.randperm(idx_g.numel(), device=device)[:int(self.spots_per_graph)]
                    keep.append(idx_g[perm])
            idx = torch.cat(keep, dim=0) if keep else torch.empty(0, dtype=torch.long, device=device)

        if idx.numel() == 0:
            return idx

        if self.spots_per_cancer > 0:
            keep = []
            cancer_sel = cancer_dom[idx]
            for k in sorted(int(v) for v in torch.unique(cancer_sel.detach()).tolist()):
                local = torch.where(cancer_sel == k)[0]
                idx_k = idx[local]
                if idx_k.numel() <= int(self.spots_per_cancer):
                    keep.append(idx_k)
                else:
                    perm = torch.randperm(idx_k.numel(), device=device)[:int(self.spots_per_cancer)]
                    keep.append(idx_k[perm])
            idx = torch.cat(keep, dim=0) if keep else torch.empty(0, dtype=torch.long, device=device)

        if idx.numel() == 0:
            return idx

        if self.label_balanced_sampling and y is not None:
            keep = []
            cancer_sel = cancer_dom[idx]
            y_sel = y[idx]
            for k in sorted(int(v) for v in torch.unique(cancer_sel.detach()).tolist()):
                idx_k = idx[torch.where(cancer_sel == k)[0]]
                y_k = y[idx_k]
                pos = idx_k[y_k == 1]
                neg = idx_k[y_k == 0]
                if pos.numel() > 0 and neg.numel() > 0:
                    n = min(int(pos.numel()), int(neg.numel()))
                    pos = pos[torch.randperm(pos.numel(), device=device)[:n]]
                    neg = neg[torch.randperm(neg.numel(), device=device)[:n]]
                    keep.append(torch.cat([pos, neg], dim=0))
                else:
                    labeled = idx_k[y_k >= 0]
                    keep.append(labeled if labeled.numel() > 0 else idx_k)
            idx = torch.cat(keep, dim=0) if keep else torch.empty(0, dtype=torch.long, device=device)

        return idx

    def _prepare(self, h, b, cancer_dom, graph_nodes=None, y=None):
        idx = self._select_indices(cancer_dom, graph_nodes=graph_nodes, y=y)
        if idx.numel() == 0:
            return None

        h_sel = self._normalize_latent(h[idx])
        b_sel = self._normalize_latent(b[idx])
        cancer_sel = cancer_dom[idx].long()
        y_sel = y[idx] if y is not None else None

        active = []
        for k in sorted(int(v) for v in torch.unique(cancer_sel.detach()).tolist()):
            if int((cancer_sel == k).sum().item()) >= int(self.min_spots):
                active.append(int(k))
        if len(active) < int(self.min_cancers):
            return None

        keep = torch.zeros_like(cancer_sel, dtype=torch.bool)
        for k in active:
            keep = keep | (cancer_sel == int(k))
        h_sel = h_sel[keep]
        b_sel = b_sel[keep]
        cancer_sel = cancer_sel[keep]
        y_sel = y_sel[keep] if y_sel is not None else None
        if h_sel.numel() == 0:
            return None

        return {
            'h': h_sel,
            'b': b_sel,
            'cancer': cancer_sel,
            'y': y_sel,
            'active_domains': active,
        }

    def _support_subset(self, b):
        if self.support_size is None or int(self.support_size) <= 0 or b.size(0) <= int(self.support_size):
            return b
        perm = torch.randperm(b.size(0), device=b.device)[:int(self.support_size)]
        return b[perm]

    def _regularizer(self, t):
        eps = max(float(self.epsilon), 1e-8)
        if self.regularizer == 'entropy':
            return eps * torch.exp(torch.clamp(t / eps, max=20.0))
        return F.relu(t).pow(2) / (2.0 * eps)

    def _dual_objective(self, prepared):
        h = prepared['h']
        b_support = self._support_subset(prepared['b'])
        cancer = prepared['cancer']
        active = prepared['active_domains']
        weights = h.new_full((len(active),), 1.0 / len(active))
        g_centered = self.potentials.g_centered(b_support, active, weights=weights)

        objs = []
        dim = max(int(h.size(-1)), 1)
        for row, k in enumerate(active):
            xk = h[cancer == int(k)]
            if xk.size(0) < int(self.min_spots):
                continue
            f_x = self.potentials.f_one_domain(xk, int(k))
            cost = torch.cdist(xk, b_support, p=2).pow(2) / float(dim)
            t = f_x.view(-1, 1) + g_centered[row].view(1, -1) - cost
            objs.append(f_x.mean() - self._regularizer(t).mean())

        if not objs:
            return self._zero(h, requires_grad=True)
        return torch.stack(objs).mean()

    def _euclidean_pairwise_loss(self, prepared):
        b = prepared['b']
        cancer = prepared['cancer']
        active = prepared['active_domains']
        dim_scale = math.sqrt(max(int(b.size(-1)), 1))
        losses = []
        d_qq = torch.cdist(b, b, p=2).mean() / dim_scale if b.size(0) > 1 else self._zero(b, requires_grad=True)
        for k in active:
            bk = b[cancer == int(k)]
            if bk.size(0) < int(self.min_spots):
                continue
            d_kq = torch.cdist(bk, b, p=2).mean() / dim_scale
            d_kk = torch.cdist(bk, bk, p=2).mean() / dim_scale if bk.size(0) > 1 else self._zero(b, requires_grad=True)
            losses.append(2.0 * d_kq - d_kk - d_qq)
        if not losses:
            return self._zero(b, requires_grad=True)
        return torch.stack(losses).mean()

    def _state_direction_loss(self, prepared):
        if self.state_direction is None or prepared.get('y', None) is None:
            return self._zero(prepared['b'], requires_grad=True), 0

        b = prepared['b']
        cancer = prepared['cancer']
        y = prepared['y']
        v = F.normalize(self.state_direction, dim=0)
        losses = []
        for k in prepared['active_domains']:
            mask_k = cancer == int(k)
            pos = b[mask_k & (y == 1)]
            neg = b[mask_k & (y == 0)]
            if pos.size(0) == 0 or neg.size(0) == 0:
                continue
            delta = pos.mean(dim=0) - neg.mean(dim=0)
            if torch.linalg.vector_norm(delta) <= 1e-8:
                continue
            losses.append(1.0 - F.cosine_similarity(delta.view(1, -1), v.view(1, -1), dim=1).squeeze(0))
        if not losses:
            return self._zero(b, requires_grad=True), 0
        return torch.stack(losses).mean(), len(losses)

    def _anchor_loss(self, prepared):
        return (prepared['h'] - prepared['b']).pow(2).mean()

    def potential_loss(self, h, b, cancer_dom, graph_nodes=None, y=None):
        prepared = self._prepare(h, b, cancer_dom, graph_nodes=graph_nodes, y=y)
        if prepared is None:
            return self._zero(b, requires_grad=False), self._empty_stats(valid=False)

        if self.loss_type == 'dual_potential':
            dual_obj = self._dual_objective(prepared)
            loss = -dual_obj
            stats = self._stats_from_prepared(prepared)
            stats.update({
                'valid': True,
                'wb_potential_loss': loss.detach(),
                'wb_dual_obj': dual_obj.detach(),
                'wb_euclid_pairwise': h.new_tensor(float('nan')),
            })
            return loss, stats

        pot_score = self.potentials.forward_by_domain(prepared['b'], prepared['cancer'])
        active_means = []
        for k in prepared['active_domains']:
            active_means.append(self.potentials.forward_one_domain(prepared['b'], int(k)).mean())
        constraint = torch.stack(active_means).mean().pow(2) if active_means else self._zero(prepared['b'], True)
        loss = pot_score.mean() + self.potential_constraint_weight * constraint
        stats = self._stats_from_prepared(prepared)
        stats.update({
            'valid': True,
            'wb_potential_loss': loss.detach(),
            'wb_dual_obj': h.new_tensor(float('nan')),
            'wb_euclid_pairwise': h.new_tensor(float('nan')),
            'wb_potential_score': pot_score.mean().detach(),
        })
        return loss, stats

    def model_loss(self, h, b, cancer_dom, graph_nodes=None, y=None):
        prepared = self._prepare(h, b, cancer_dom, graph_nodes=graph_nodes, y=y)
        if prepared is None:
            z = self._zero(h, requires_grad=True) + self._zero(b, requires_grad=True)
            return z, z, self._empty_stats(valid=False)

        anchor = self._anchor_loss(prepared)
        shape_loss, shape_count = self._state_direction_loss(prepared)

        if self.loss_type == 'dual_potential':
            dual_obj = self._dual_objective(prepared)
            raw_loss = dual_obj + self.state_direction_weight * shape_loss
            stats = self._stats_from_prepared(prepared)
            stats.update({
                'valid': True,
                'wb_loss': raw_loss.detach(),
                'wb_dual_obj': dual_obj.detach(),
                'wb_euclid_pairwise': h.new_tensor(float('nan')),
                'wb_anchor': anchor.detach(),
                'wb_state_direction': shape_loss.detach(),
                'wb_state_direction_count': float(shape_count),
            })
            return raw_loss, anchor, stats

        pair_loss = self._euclidean_pairwise_loss(prepared)
        pot_score = self.potentials.forward_by_domain(prepared['b'], prepared['cancer'])
        pot_main = -pot_score.mean()
        raw_loss = (
            self.euclid_pairwise_weight * pair_loss
            + self.potential_weight * pot_main
            + self.state_direction_weight * shape_loss
        )
        stats = self._stats_from_prepared(prepared)
        stats.update({
            'valid': True,
            'wb_loss': raw_loss.detach(),
            'wb_dual_obj': h.new_tensor(float('nan')),
            'wb_euclid_pairwise': pair_loss.detach(),
            'wb_anchor': anchor.detach(),
            'wb_state_direction': shape_loss.detach(),
            'wb_state_direction_count': float(shape_count),
            'wb_potential_score': pot_score.mean().detach(),
        })
        return raw_loss, anchor, stats

    def _stats_from_prepared(self, prepared):
        return {
            'wb_active_cancers': float(len(prepared['active_domains'])),
            'wb_active_spots': float(prepared['h'].size(0)),
        }

    def _empty_stats(self, valid=False):
        return {
            'valid': bool(valid),
            'wb_active_cancers': 0.0,
            'wb_active_spots': 0.0,
        }
