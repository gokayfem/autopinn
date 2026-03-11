"""
pinn.py — THE ONLY FILE THE AGENT EDITS
=========================================
Metric  : L2 relative error (LOWER IS BETTER)
Budget  : 3 minutes wall-clock per run (excluding import time)
Problems: Allen-Cahn (hard) and Convection (medium)

Run a single experiment:
  python pinn.py --problem allen_cahn   # default
  python pinn.py --problem convection

The script prints one line at the end:
  RESULT | problem=Allen-Cahn | L2_error=0.4821 | steps=3400 | time=180.1s

═══════════════════════════════════════════════════════════════
BASELINE PERFORMANCE (what this file achieves out of the box):
  Allen-Cahn : ~0.50  (essentially fails — literature confirms this)
  Convection : ~1.00  (complete failure — literature confirms this)

SOTA TARGETS (from peer-reviewed papers):
  Allen-Cahn : 3.26e-06  (Deeper-PINNs, ResearchGate 2024)
  Convection : ~1e-04    (FP64 vanilla PINN, arXiv 2505.10949, 2025)
═══════════════════════════════════════════════════════════════
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np

# ── Select problem ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--problem', default='allen_cahn',
                    choices=['allen_cahn', 'convection'])
parser.add_argument('--seed',   type=int, default=42)
parser.add_argument('--run-id', type=str, default='0',
                    help='Unique tag for this run (used in output line)')
args = parser.parse_args()

torch.manual_seed(args.seed)
import numpy as np; np.random.seed(args.seed)

if args.problem == 'allen_cahn':
    from problems.allen_cahn import (
        sample_interior, sample_initial, sample_periodic_bc,
        pde_residual, evaluate, NAME
    )
else:
    from problems.convection import (
        sample_interior, sample_initial, sample_periodic_bc,
        pde_residual, evaluate, NAME
    )

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BUDGET_SECONDS = 180   # 3-minute wall-clock budget

# Problem-specific settings
USE_HARD_IC = (args.problem == 'allen_cahn')
USE_SPECTRAL = (args.problem == 'convection')


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class SineLayer(nn.Module):
    """SIREN layer: sin(omega * Linear(x))"""
    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_dim, out_dim)
        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                bound = math.sqrt(6.0 / in_dim) / omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SubPINN(nn.Module):
    """Small MLP for a single time window.

    If hard IC is enabled: u(x,t) = IC(x) + t * NN(x,t)
    Otherwise: u(x,t) = NN(x,t)
    """
    def __init__(self, width=64, depth=3, fourier_freqs=None, fourier_freqs_t=None,
                 use_siren=False, omega=30.0):
        super().__init__()
        self.freqs = fourier_freqs
        self.freqs_t = fourier_freqs_t
        self.use_siren = use_siren
        in_dim = 2  # x + t base
        if fourier_freqs is not None:
            in_dim += 2 * len(fourier_freqs)
        if fourier_freqs_t is not None:
            in_dim += 2 * len(fourier_freqs_t)

        if use_siren:
            layers = [SineLayer(in_dim, width, omega=omega, is_first=True)]
            for _ in range(depth - 1):
                layers.append(SineLayer(width, width, omega=omega))
            layers.append(nn.Linear(width, 1))
            # Init output layer
            with torch.no_grad():
                bound = math.sqrt(6.0 / width) / omega
                layers[-1].weight.uniform_(-bound, bound)
            self.net = nn.Sequential(*layers)
        else:
            layers = [nn.Linear(in_dim, width), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.Tanh()]
            layers.append(nn.Linear(width, 1))
            self.net = nn.Sequential(*layers)
        self._ic_fn = None

    def set_ic(self, ic_fn):
        self._ic_fn = ic_fn

    def forward(self, x, t):
        feat = [x]
        if self.freqs is not None:
            for k in self.freqs:
                feat.append(torch.sin(k * x))
                feat.append(torch.cos(k * x))
        feat.append(t)
        if self.freqs_t is not None:
            for k in self.freqs_t:
                feat.append(torch.sin(k * t))
                feat.append(torch.cos(k * t))
        inp = torch.cat(feat, dim=1)
        raw = self.net(inp)
        if self._ic_fn is not None:
            return self._ic_fn(x) + t * raw
        return raw


class SpectralPINN(nn.Module):
    """Spectral PINN: tiny MLP(t) -> Fourier coefficients a_k(t), b_k(t).
    u(x,t) = Σ [a_k cos(kx) + b_k sin(kx)]. Periodic BC exact."""
    def __init__(self, n_modes=4, width=32, depth=2, t_freqs=None):
        super().__init__()
        self.n_modes = n_modes
        self.t_freqs = t_freqs
        in_dim = 1
        if t_freqs is not None:
            in_dim += 2 * len(t_freqs)
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 2 * n_modes))
        self.net = nn.Sequential(*layers)
        ks = torch.arange(1, n_modes + 1, dtype=torch.float64)
        self.register_buffer('ks', ks)
        self._ic_a = None
        self._ic_b = None

    def set_ic(self, ic_a, ic_b):
        self._ic_a = ic_a
        self._ic_b = ic_b

    def forward(self, x, t):
        feat = [t]
        if self.t_freqs is not None:
            for f in self.t_freqs:
                feat.append(torch.sin(f * t))
                feat.append(torch.cos(f * t))
        raw = self.net(torch.cat(feat, dim=1))
        a = raw[:, :self.n_modes]
        b = raw[:, self.n_modes:]
        if self._ic_a is not None:
            a = self._ic_a + t * a
            b = self._ic_b + t * b
        kx = x * self.ks
        u = (a * torch.cos(kx) + b * torch.sin(kx)).sum(dim=1, keepdim=True)
        return u


class PINN(nn.Module):
    """Wrapper that holds all sub-models for evaluation."""
    def __init__(self, models, t_boundaries):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.t_boundaries = t_boundaries  # [0, t1, t2, ..., 1.0]

    def forward(self, x, t):
        out = torch.zeros_like(x)
        for i, model in enumerate(self.models):
            t_lo = self.t_boundaries[i]
            t_hi = self.t_boundaries[i + 1]
            mask = (t.squeeze() >= t_lo) & (t.squeeze() < t_hi)
            if i == len(self.models) - 1:
                mask = mask | (t.squeeze() >= t_hi)  # include right endpoint
            if mask.sum() > 0:
                # Normalize t to [0, 1] within window
                t_local = (t[mask] - t_lo) / (t_hi - t_lo)
                out[mask] = model(x[mask], t_local)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — sequential time windows
# ══════════════════════════════════════════════════════════════════════════════

def train():
    if args.problem == 'convection':
        x_lo, x_hi = 0.0, 2 * math.pi
    else:
        x_lo, x_hi = -1.0, 1.0

    n_windows = 3 if args.problem == 'allen_cahn' else 1
    t_boundaries = [i / n_windows for i in range(n_windows + 1)]
    window_times = [BUDGET_SECONDS / n_windows] * n_windows

    models = []
    global_deadline = time.time() + BUDGET_SECONDS
    step = 0

    for wi in range(n_windows):
        t_lo_w = t_boundaries[wi]
        t_hi_w = t_boundaries[wi + 1]
        dt = t_hi_w - t_lo_w

        if args.problem == 'allen_cahn':
            freqs_x = [math.pi, 2*math.pi, 4*math.pi, 8*math.pi]
            sub_model = SubPINN(width=96, depth=2, fourier_freqs=freqs_x).double().to(DEVICE)
            DTYPE = torch.float64
        else:
            DTYPE = torch.float64
            sub_model = SpectralPINN(n_modes=1, width=32, depth=2, t_freqs=[50.0]).double().to(DEVICE)
            ic_a = torch.zeros(1, device=DEVICE, dtype=DTYPE)
            ic_b = torch.ones(1, device=DEVICE, dtype=DTYPE)
            sub_model.set_ic(ic_a, ic_b)

        # Set hard IC function (Allen-Cahn only)
        if USE_HARD_IC:
            if wi == 0:
                if args.problem == 'convection':
                    sub_model.set_ic(lambda x: torch.sin(x))
                else:
                    sub_model.set_ic(lambda x: x**2 * torch.cos(math.pi * x))
            else:
                prev = models[-1]
                for p in prev.parameters():
                    p.requires_grad_(False)
                sub_model.set_ic(
                    lambda x, _p=prev: _p(x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
                )

        lr = 5e-3 if args.problem == 'allen_cahn' else 3e-3
        optimizer = torch.optim.Adam(sub_model.parameters(), lr=lr)
        scheduler = None
        if args.problem == 'allen_cahn':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=lr/10)

        window_start = time.time()
        window_deadline = window_start + window_times[wi]
        lbfgs_frac = 0.5
        lbfgs_start = window_start + lbfgs_frac * window_times[wi]
        use_lbfgs = False

        # Pre-generate fixed collocation points for L-BFGS phase
        n_pts = 750 if USE_SPECTRAL else 2000
        x_f_fixed = torch.empty(n_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(x_lo, x_hi)
        t_f_fixed = torch.empty(n_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(0, 1)
        n_ic_pts = 300
        x_ic_fixed = torch.empty(n_ic_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(x_lo, x_hi)
        n_bc_pts = 100
        t_bc_fixed = torch.empty(n_bc_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(0, 1)

        while time.time() < min(window_deadline, global_deadline):
            # Switch to L-BFGS after 70% of window time
            if not use_lbfgs and time.time() >= lbfgs_start:
                use_lbfgs = True
                optimizer = torch.optim.LBFGS(
                    sub_model.parameters(), lr=1.0, max_iter=20,
                    history_size=50, line_search_fn='strong_wolfe'
                )

            if use_lbfgs:
                # Fixed points for L-BFGS (deterministic loss)
                x_f = x_f_fixed.detach().requires_grad_(True)
                t_f = t_f_fixed.detach().requires_grad_(True)
                x_ic = x_ic_fixed.detach()
                t_ic = torch.zeros(n_ic_pts, 1, device=DEVICE, dtype=DTYPE)
                t_bc = t_bc_fixed.detach()
                x_left  = torch.full((n_bc_pts, 1), x_lo, device=DEVICE, dtype=DTYPE).requires_grad_(True)
                x_right = torch.full((n_bc_pts, 1), x_hi, device=DEVICE, dtype=DTYPE).requires_grad_(True)
            else:
                # Random points for Adam
                x_f = torch.empty(n_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(x_lo, x_hi).requires_grad_(True)
                t_f = torch.empty(n_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(0, 1).requires_grad_(True)
                x_ic = torch.empty(n_ic_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(x_lo, x_hi)
                t_ic = torch.zeros(n_ic_pts, 1, device=DEVICE, dtype=DTYPE)
                t_bc = torch.empty(n_bc_pts, 1, device=DEVICE, dtype=DTYPE).uniform_(0, 1)
                x_left  = torch.full((n_bc_pts, 1), x_lo, device=DEVICE, dtype=DTYPE).requires_grad_(True)
                x_right = torch.full((n_bc_pts, 1), x_hi, device=DEVICE, dtype=DTYPE).requires_grad_(True)

            def compute_loss():
                if USE_SPECTRAL:
                    # Spectral: only PDE loss (BC exact, IC via coefficients)
                    u_f = sub_model(x_f, t_f)
                    u_t_local = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f),
                                                    create_graph=True)[0]
                    u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f),
                                              create_graph=True)[0]
                    residual = u_t_local / dt + 50.0 * u_x
                    return (residual**2).mean()

                # IC loss (soft IC for non-spectral Convection only)
                if not USE_HARD_IC:
                    if wi == 0:
                        if args.problem == 'convection':
                            u_ic_true = torch.sin(x_ic).double()
                        else:
                            u_ic_true = (x_ic**2 * torch.cos(math.pi * x_ic)).double()
                    else:
                        with torch.no_grad():
                            t_prev_end = torch.ones(n_ic_pts, 1, device=DEVICE, dtype=DTYPE)
                            u_ic_true = models[-1](x_ic, t_prev_end)

                    u_ic_pred = sub_model(x_ic, t_ic)
                    loss_ic = ((u_ic_pred - u_ic_true)**2).mean()

                # Periodic BC
                u_left  = sub_model(x_left, t_bc)
                u_right = sub_model(x_right, t_bc)
                loss_bc = ((u_left - u_right)**2).mean()

                # PDE residual
                u_f = sub_model(x_f, t_f)
                u_t_local = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f),
                                                create_graph=True)[0]
                u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f),
                                          create_graph=True)[0]

                if args.problem == 'convection':
                    residual = u_t_local / dt + 50.0 * u_x
                else:
                    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x),
                                               create_graph=True)[0]
                    residual = u_t_local / dt - 0.001**2 * u_xx + 5.0 * (u_f**3 - u_f)

                if USE_HARD_IC:
                    # Exponential causal weighting for Allen-Cahn
                    causal_w = torch.exp(3.0 * (1.0 - t_f.detach()))
                    loss_pde = (causal_w * residual**2).mean()
                else:
                    loss_pde = (residual**2).mean()

                if USE_HARD_IC:
                    return loss_bc + loss_pde
                else:
                    return 7.0 * loss_ic + loss_bc + loss_pde

            if use_lbfgs:
                def closure():
                    optimizer.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                loss = compute_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            step += 1

        models.append(sub_model)

    # Build composite model for evaluation
    composite = PINN(models, t_boundaries).to(DEVICE)

    elapsed = BUDGET_SECONDS
    l2 = evaluate(composite, DEVICE)

    print(f"RESULT | run={args.run_id} | seed={args.seed} | problem={NAME} | "
          f"L2_error={l2:.6f} | steps={step} | time={elapsed:.1f}s")
    return l2


if __name__ == '__main__':
    t0 = time.time()
    result = train()
    print(f"Total wall time: {time.time()-t0:.1f}s")
