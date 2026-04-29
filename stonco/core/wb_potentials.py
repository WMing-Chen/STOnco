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


class PriorSupportGenerator(nn.Module):
    """Generator for an independent shared barycenter support sample b = G(z)."""

    def __init__(self, in_dim, prior_dim=128, hidden=128, dropout=0.0, prior_type='normal'):
        super().__init__()
        self.in_dim = int(in_dim)
        self.prior_dim = int(prior_dim)
        self.prior_type = str(prior_type).lower()
        if self.prior_dim <= 0:
            raise ValueError(f'wb_prior_dim must be > 0, got: {prior_dim}')
        if self.prior_type not in {'normal', 'uniform'}:
            raise ValueError(f"wb_prior_type must be normal or uniform, got: {prior_type}")
        self.net = nn.Sequential(
            nn.Linear(self.prior_dim, int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.in_dim),
        )

    def sample_prior(self, n, device=None, dtype=None, generator=None):
        n = int(n)
        if n <= 0:
            raise ValueError(f'prior support sample size must be > 0, got: {n}')
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        if self.prior_type == 'uniform':
            return torch.empty(n, self.prior_dim, device=device, dtype=dtype).uniform_(-1.0, 1.0, generator=generator)
        return torch.randn(n, self.prior_dim, device=device, dtype=dtype, generator=generator)

    def forward(self, z):
        return self.net(z)

    def sample(self, n, device=None, dtype=None, generator=None):
        z = self.sample_prior(n, device=device, dtype=dtype, generator=generator)
        return self.forward(z)


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
        sinkhorn_iters=50,
        sw_num_projections=64,
        support_mode='generated_support',
        mmd_num_kernels=5,
        mmd_kernel_mul=2.0,
        mmd_sigma=None,
    ):
        super().__init__()
        self.n_domains = int(n_domains)
        self.in_dim = int(in_dim)
        self.loss_type = str(loss_type).lower()
        self.support_mode = str(support_mode).lower()
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
        self.sinkhorn_iters = max(1, int(sinkhorn_iters))
        self.sw_num_projections = max(1, int(sw_num_projections))
        self.mmd_num_kernels = max(1, int(mmd_num_kernels))
        self.mmd_kernel_mul = float(mmd_kernel_mul)
        self.mmd_sigma = None if mmd_sigma is None else float(mmd_sigma)

        if self.support_mode not in {'generated_support', 'prior_generator'}:
            raise ValueError(f"wb_support_mode must be generated_support or prior_generator, got: {support_mode}")
        if self.support_mode == 'prior_generator' and self.loss_type not in {'sinkhorn_divergence', 'sliced_wasserstein', 'mmd'}:
            raise ValueError(
                "prior_generator supports only sinkhorn_divergence, sliced_wasserstein, or mmd, "
                f"got: {loss_type}"
            )
        if self.loss_type == 'mmd' and self.support_mode != 'prior_generator':
            raise ValueError("wb_loss_type=mmd is only supported when wb_support_mode=prior_generator")

        if self.loss_type == 'euclidean_pairwise':
            self.potentials = SinglePotentialBank(self.n_domains, self.in_dim, hidden=potential_hidden)
        elif self.loss_type == 'dual_potential':
            self.potentials = DualPotentialBank(self.n_domains, self.in_dim, hidden=potential_hidden)
        elif self.loss_type in {'sinkhorn_divergence', 'sliced_wasserstein', 'mmd'}:
            self.potentials = None
        else:
            raise ValueError(
                "wb_loss_type must be dual_potential, euclidean_pairwise, "
                f"sinkhorn_divergence, sliced_wasserstein, or mmd, got: {loss_type}"
            )

        if self.regularizer not in {'l2', 'entropy'}:
            raise ValueError(f"wb_regularizer must be l2 or entropy, got: {regularizer}")

        self.state_direction = None
        if bool(state_direction):
            v = torch.empty(self.in_dim)
            nn.init.normal_(v, mean=0.0, std=1.0 / math.sqrt(max(self.in_dim, 1)))
            self.state_direction = nn.Parameter(v)

    def potential_parameters(self):
        if self.potentials is None:
            return []
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

    def _select_indices(self, cancer_dom, graph_nodes=None, y=None, generator=None):
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
                    perm = torch.randperm(idx_g.numel(), device=device, generator=generator)[:int(self.spots_per_graph)]
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
                    perm = torch.randperm(idx_k.numel(), device=device, generator=generator)[:int(self.spots_per_cancer)]
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
                    pos = pos[torch.randperm(pos.numel(), device=device, generator=generator)[:n]]
                    neg = neg[torch.randperm(neg.numel(), device=device, generator=generator)[:n]]
                    keep.append(torch.cat([pos, neg], dim=0))
                else:
                    labeled = idx_k[y_k >= 0]
                    keep.append(labeled if labeled.numel() > 0 else idx_k)
            idx = torch.cat(keep, dim=0) if keep else torch.empty(0, dtype=torch.long, device=device)

        return idx

    def _prepare(self, h, b, cancer_dom, graph_nodes=None, y=None, generator=None):
        idx = self._select_indices(cancer_dom, graph_nodes=graph_nodes, y=y, generator=generator)
        if idx.numel() == 0:
            return None

        h_sel = self._normalize_latent(h[idx])
        if self.support_mode == 'prior_generator':
            b_sel = self._normalize_latent(b)
        else:
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
        if self.support_mode != 'prior_generator':
            b_sel = b_sel[keep]
        cancer_sel = cancer_sel[keep]
        y_sel = y_sel[keep] if y_sel is not None else None
        if h_sel.numel() == 0 or b_sel.numel() == 0:
            return None

        return {
            'h': h_sel,
            'b': b_sel,
            'cancer': cancer_sel,
            'y': y_sel,
            'active_domains': active,
        }

    def _support_subset(self, b, generator=None):
        if self.support_size is None or int(self.support_size) <= 0 or b.size(0) <= int(self.support_size):
            return b
        perm = torch.randperm(b.size(0), device=b.device, generator=generator)[:int(self.support_size)]
        return b[perm]

    def _sample_projection_directions(self, ref, generator=None):
        directions = torch.randn(
            int(self.sw_num_projections),
            int(self.in_dim),
            device=ref.device,
            dtype=ref.dtype,
            generator=generator,
        )
        return F.normalize(directions, p=2, dim=1, eps=1e-12)

    def _wasserstein_1d_weighted(self, x, y, weights_x=None, weights_y=None):
        ref = x if x.numel() > 0 else y
        if x.numel() == 0 or y.numel() == 0:
            return self._zero(ref, requires_grad=True)

        x_sorted, x_perm = torch.sort(x.reshape(-1))
        y_sorted, y_perm = torch.sort(y.reshape(-1))
        if weights_x is None and weights_y is None and x_sorted.numel() == y_sorted.numel():
            return (x_sorted - y_sorted).pow(2).mean()

        def _prepare_weights(weights, perm, n, ref_tensor):
            if weights is None:
                out = ref_tensor.new_full((n,), 1.0 / float(max(n, 1)))
            else:
                out = weights.reshape(-1).to(device=ref_tensor.device, dtype=ref_tensor.dtype)
                if out.numel() != n:
                    raise ValueError(f'1D Wasserstein weights size mismatch: expected {n}, got {out.numel()}')
                out = out[perm]
                total = out.sum().clamp_min(1e-12)
                out = out / total
            return out

        wx = _prepare_weights(weights_x, x_perm, x_sorted.numel(), x_sorted)
        wy = _prepare_weights(weights_y, y_perm, y_sorted.numel(), y_sorted)
        cdf_x = wx.cumsum(dim=0)
        cdf_y = wy.cumsum(dim=0)
        breaks = torch.unique(torch.cat([cdf_x, cdf_y], dim=0), sorted=True)
        breaks = torch.cat([breaks.new_zeros(1), breaks], dim=0)
        delta = breaks[1:] - breaks[:-1]
        idx_x = torch.searchsorted(cdf_x, breaks[1:], right=False).clamp(max=x_sorted.numel() - 1)
        idx_y = torch.searchsorted(cdf_y, breaks[1:], right=False).clamp(max=y_sorted.numel() - 1)
        return torch.sum(delta * (x_sorted[idx_x] - y_sorted[idx_y]).pow(2))

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

    def _sinkhorn_ot(self, x, y, return_plan=False):
        if x.size(0) == 0 or y.size(0) == 0:
            z = self._zero(x if x.numel() > 0 else y, requires_grad=True)
            return (z, None) if return_plan else z

        eps = max(float(self.epsilon), 1e-6)
        nx = int(x.size(0))
        ny = int(y.size(0))
        dim = max(int(x.size(-1)), 1)
        cost = torch.cdist(x, y, p=2).pow(2) / float(dim)

        log_a = x.new_full((nx,), -math.log(nx))
        log_b = y.new_full((ny,), -math.log(ny))
        log_k = -cost / eps

        log_u = x.new_zeros(nx)
        log_v = y.new_zeros(ny)
        for _ in range(int(self.sinkhorn_iters)):
            log_u = log_a - torch.logsumexp(log_k + log_v.view(1, -1), dim=1)
            log_v = log_b - torch.logsumexp(log_k + log_u.view(-1, 1), dim=0)

        log_pi = log_u.view(-1, 1) + log_k + log_v.view(1, -1)
        pi = torch.exp(log_pi)
        kl_term = log_pi - log_a.view(-1, 1) - log_b.view(1, -1)
        ot_value = (pi * cost).sum() + eps * (pi * kl_term).sum()
        if return_plan:
            return ot_value, pi
        return ot_value

    def _sinkhorn_divergence_loss(self, prepared):
        h = prepared['h']
        b_support = self._support_subset(prepared['b'])
        cancer = prepared['cancer']
        active = prepared['active_domains']
        if b_support.size(0) < int(self.min_spots):
            return self._zero(h, requires_grad=True)

        ot_qq = self._sinkhorn_ot(b_support, b_support)
        losses = []
        for k in active:
            xk = h[cancer == int(k)]
            if xk.size(0) < int(self.min_spots):
                continue
            ot_kq = self._sinkhorn_ot(xk, b_support)
            ot_kk = self._sinkhorn_ot(xk, xk)
            losses.append(ot_kq - 0.5 * ot_kk - 0.5 * ot_qq)
        if not losses:
            return self._zero(h, requires_grad=True)
        return torch.stack(losses).mean()

    def _build_sigma_list(self, x, y):
        if self.mmd_sigma is not None:
            base_sigma = torch.tensor(float(self.mmd_sigma), dtype=x.dtype, device=x.device)
        else:
            z = torch.cat([x, y], dim=0)
            if z.size(0) <= 1:
                base_sigma = torch.tensor(1.0, dtype=x.dtype, device=x.device)
            else:
                sq = torch.cdist(z, z, p=2).pow(2)
                mask = ~torch.eye(sq.size(0), dtype=torch.bool, device=sq.device)
                vals = sq[mask]
                base_sigma = torch.sqrt(vals.mean().clamp_min(1e-12)) if vals.numel() > 0 else torch.tensor(1.0, dtype=x.dtype, device=x.device)
        center = int(self.mmd_num_kernels) // 2
        return [
            (base_sigma * (float(self.mmd_kernel_mul) ** (i - center))).clamp_min(1e-6)
            for i in range(int(self.mmd_num_kernels))
        ]

    def _rbf_kernel(self, x, y, sigma_list):
        sq = torch.cdist(x, y, p=2).pow(2)
        k = torch.zeros_like(sq)
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma.pow(2).clamp_min(1e-12))
            k = k + torch.exp(-sq * gamma)
        return k

    def _mmd2_unbiased(self, x, y, sigma_list):
        n = int(x.size(0))
        m = int(y.size(0))
        if n < 2 or m < 2:
            return self._zero(x if x.numel() > 0 else y, requires_grad=True)
        k_xx = self._rbf_kernel(x, x, sigma_list)
        k_yy = self._rbf_kernel(y, y, sigma_list)
        k_xy = self._rbf_kernel(x, y, sigma_list)
        sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / float(n * (n - 1))
        sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / float(m * (m - 1))
        return torch.clamp(sum_xx + sum_yy - 2.0 * k_xy.mean(), min=0.0)

    def _mmd_barycenter_loss(self, prepared, generator=None):
        h = prepared['h']
        b_support = self._support_subset(prepared['b'], generator=generator)
        cancer = prepared['cancer']
        active = prepared['active_domains']
        losses = []
        for k in active:
            xk = h[cancer == int(k)]
            if xk.size(0) < int(self.min_spots) or b_support.size(0) < 2:
                continue
            sigma_list = self._build_sigma_list(xk, b_support)
            losses.append(self._mmd2_unbiased(xk, b_support, sigma_list))
        if not losses:
            return self._zero(h, requires_grad=True)
        return torch.stack(losses).mean()

    def _sliced_wasserstein_loss(self, prepared, generator=None):
        h = prepared['h']
        b_support = self._support_subset(prepared['b'], generator=generator)
        if b_support.size(0) < 2:
            return None

        cancer = prepared['cancer']
        active = prepared['active_domains']
        directions = self._sample_projection_directions(b_support, generator=generator)
        proj_support = torch.matmul(b_support, directions.t())

        losses = []
        for k in active:
            xk = h[cancer == int(k)]
            if xk.size(0) < int(self.min_spots):
                continue
            proj_xk = torch.matmul(xk, directions.t())
            dir_losses = [
                self._wasserstein_1d_weighted(proj_xk[:, ell], proj_support[:, ell])
                for ell in range(int(directions.size(0)))
            ]
            losses.append(torch.stack(dir_losses).mean())

        if not losses:
            return self._zero(h, requires_grad=True)
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

    def potential_loss(self, h, b, cancer_dom, graph_nodes=None, y=None, generator=None):
        if self.potentials is None:
            return self._zero(b, requires_grad=False), self._empty_stats(valid=False)

        prepared = self._prepare(h, b, cancer_dom, graph_nodes=graph_nodes, y=y, generator=generator)
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
                'wb_sinkhorn': h.new_tensor(float('nan')),
                'wb_sliced_wasserstein': h.new_tensor(float('nan')),
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
            'wb_sinkhorn': h.new_tensor(float('nan')),
            'wb_sliced_wasserstein': h.new_tensor(float('nan')),
            'wb_potential_score': pot_score.mean().detach(),
        })
        return loss, stats

    def model_loss(self, h, b, cancer_dom, graph_nodes=None, y=None, generator=None):
        prepared = self._prepare(h, b, cancer_dom, graph_nodes=graph_nodes, y=y, generator=generator)
        if prepared is None:
            z = self._zero(h, requires_grad=True) + self._zero(b, requires_grad=True)
            return z, z, self._empty_stats(valid=False)

        if self.support_mode == 'prior_generator':
            shape_loss = self._zero(prepared['h'], requires_grad=True)
            shape_count = 0
        else:
            shape_loss, shape_count = self._state_direction_loss(prepared)

        if self.support_mode == 'prior_generator':
            anchor = self._zero(prepared['b'], requires_grad=True)
            if self.loss_type == 'sinkhorn_divergence':
                bary_loss = self._sinkhorn_divergence_loss(prepared)
                sinkhorn_stat = bary_loss.detach()
                sliced_stat = h.new_tensor(float('nan'))
                mmd_stat = h.new_tensor(float('nan'))
            elif self.loss_type == 'sliced_wasserstein':
                bary_loss = self._sliced_wasserstein_loss(prepared, generator=generator)
                if bary_loss is None:
                    z = self._zero(h, requires_grad=True) + self._zero(b, requires_grad=True)
                    return z, anchor, self._empty_stats(valid=False)
                sinkhorn_stat = h.new_tensor(float('nan'))
                sliced_stat = bary_loss.detach()
                mmd_stat = h.new_tensor(float('nan'))
            elif self.loss_type == 'mmd':
                bary_loss = self._mmd_barycenter_loss(prepared, generator=generator)
                sinkhorn_stat = h.new_tensor(float('nan'))
                sliced_stat = h.new_tensor(float('nan'))
                mmd_stat = bary_loss.detach()
            else:
                raise ValueError(f'Unsupported prior_generator loss_type: {self.loss_type}')
            raw_loss = bary_loss
            stats = self._stats_from_prepared(prepared)
            stats.update({
                'valid': True,
                'wb_loss': raw_loss.detach(),
                'wb_dual_obj': h.new_tensor(float('nan')),
                'wb_euclid_pairwise': h.new_tensor(float('nan')),
                'wb_sinkhorn': sinkhorn_stat,
                'wb_sliced_wasserstein': sliced_stat,
                'wb_mmd': mmd_stat,
                'wb_anchor': anchor.detach(),
                'wb_state_direction': shape_loss.detach(),
                'wb_state_direction_count': float(shape_count),
            })
            return raw_loss, anchor, stats

        if self.loss_type == 'dual_potential':
            anchor = self._anchor_loss(prepared)
            dual_obj = self._dual_objective(prepared)
            raw_loss = dual_obj + self.state_direction_weight * shape_loss
            stats = self._stats_from_prepared(prepared)
            stats.update({
                'valid': True,
                'wb_loss': raw_loss.detach(),
                'wb_dual_obj': dual_obj.detach(),
                'wb_euclid_pairwise': h.new_tensor(float('nan')),
                'wb_sinkhorn': h.new_tensor(float('nan')),
                'wb_sliced_wasserstein': h.new_tensor(float('nan')),
                'wb_mmd': h.new_tensor(float('nan')),
                'wb_anchor': anchor.detach(),
                'wb_state_direction': shape_loss.detach(),
                'wb_state_direction_count': float(shape_count),
            })
            return raw_loss, anchor, stats

        if self.loss_type == 'sinkhorn_divergence':
            anchor = self._anchor_loss(prepared)
            sinkhorn_loss = self._sinkhorn_divergence_loss(prepared)
            raw_loss = sinkhorn_loss + self.state_direction_weight * shape_loss
            stats = self._stats_from_prepared(prepared)
            stats.update({
                'valid': True,
                'wb_loss': raw_loss.detach(),
                'wb_dual_obj': h.new_tensor(float('nan')),
                'wb_euclid_pairwise': h.new_tensor(float('nan')),
                'wb_sinkhorn': sinkhorn_loss.detach(),
                'wb_sliced_wasserstein': h.new_tensor(float('nan')),
                'wb_mmd': h.new_tensor(float('nan')),
                'wb_anchor': anchor.detach(),
                'wb_state_direction': shape_loss.detach(),
                'wb_state_direction_count': float(shape_count),
            })
            return raw_loss, anchor, stats

        if self.loss_type == 'sliced_wasserstein':
            sliced_loss = self._sliced_wasserstein_loss(prepared, generator=generator)
            if sliced_loss is None:
                z = self._zero(h, requires_grad=True) + self._zero(b, requires_grad=True)
                return z, z, self._empty_stats(valid=False)
            anchor = self._anchor_loss(prepared)
            raw_loss = sliced_loss + self.state_direction_weight * shape_loss
            stats = self._stats_from_prepared(prepared)
            stats.update({
                'valid': True,
                'wb_loss': raw_loss.detach(),
                'wb_dual_obj': h.new_tensor(float('nan')),
                'wb_euclid_pairwise': h.new_tensor(float('nan')),
                'wb_sinkhorn': h.new_tensor(float('nan')),
                'wb_sliced_wasserstein': sliced_loss.detach(),
                'wb_mmd': h.new_tensor(float('nan')),
                'wb_anchor': anchor.detach(),
                'wb_state_direction': shape_loss.detach(),
                'wb_state_direction_count': float(shape_count),
            })
            return raw_loss, anchor, stats

        anchor = self._anchor_loss(prepared)
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
            'wb_sinkhorn': h.new_tensor(float('nan')),
            'wb_sliced_wasserstein': h.new_tensor(float('nan')),
            'wb_mmd': h.new_tensor(float('nan')),
            'wb_anchor': anchor.detach(),
            'wb_state_direction': shape_loss.detach(),
            'wb_state_direction_count': float(shape_count),
            'wb_potential_score': pot_score.mean().detach(),
        })
        return raw_loss, anchor, stats

    def _stats_from_prepared(self, prepared):
        h = prepared['h']
        b = prepared['b']
        h_std = h.std(dim=0, unbiased=False)
        b_std = b.std(dim=0, unbiased=False)
        return {
            'wb_active_cancers': float(len(prepared['active_domains'])),
            'wb_active_spots': float(h.size(0)),
            'wb_support_norm': b.norm(dim=1).mean().detach(),
            'wb_support_std': b_std.mean().detach(),
            'wb_h_norm': h.norm(dim=1).mean().detach(),
            'wb_h_std': h_std.mean().detach(),
            'wb_mean_gap': (h.mean(dim=0) - b.mean(dim=0)).norm().detach(),
            'wb_std_gap': (h_std - b_std).norm().detach(),
        }

    def _empty_stats(self, valid=False):
        return {
            'valid': bool(valid),
            'wb_active_cancers': 0.0,
            'wb_active_spots': 0.0,
            'wb_support_norm': float('nan'),
            'wb_support_std': float('nan'),
            'wb_h_norm': float('nan'),
            'wb_h_std': float('nan'),
            'wb_mean_gap': float('nan'),
            'wb_std_gap': float('nan'),
        }
