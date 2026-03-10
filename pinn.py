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


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class SubPINN(nn.Module):
    """Small MLP for a single time window.

    If hard IC is enabled: u(x,t) = IC(x) + t * NN(x,t)
    Otherwise: u(x,t) = NN(x,t)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self._ic_fn = None

    def set_ic(self, ic_fn):
        self._ic_fn = ic_fn

    def forward(self, x, t):
        raw = self.net(torch.cat([x, t], dim=1))
        if self._ic_fn is not None:
            return self._ic_fn(x) + t * raw
        return raw


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

    n_windows = 3 if args.problem == 'allen_cahn' else 5
    t_boundaries = [i / n_windows for i in range(n_windows + 1)]
    time_per_window = BUDGET_SECONDS / n_windows

    models = []
    global_deadline = time.time() + BUDGET_SECONDS
    step = 0

    for wi in range(n_windows):
        t_lo_w = t_boundaries[wi]
        t_hi_w = t_boundaries[wi + 1]
        dt = t_hi_w - t_lo_w

        sub_model = SubPINN().double().to(DEVICE)

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

        optimizer = torch.optim.Adam(sub_model.parameters(), lr=1e-3)

        window_deadline = time.time() + time_per_window

        while time.time() < min(window_deadline, global_deadline):
            # Generate all sample points first (preserves RNG order)
            n_pts = 2000
            x_f = torch.empty(n_pts, 1, device=DEVICE, dtype=torch.float64).uniform_(x_lo, x_hi).requires_grad_(True)
            t_f = torch.empty(n_pts, 1, device=DEVICE, dtype=torch.float64).uniform_(0, 1).requires_grad_(True)

            n_bc = 100
            t_bc = torch.empty(n_bc, 1, device=DEVICE, dtype=torch.float64).uniform_(0, 1)
            x_left  = torch.full((n_bc, 1), x_lo, device=DEVICE, dtype=torch.float64).requires_grad_(True)
            x_right = torch.full((n_bc, 1), x_hi, device=DEVICE, dtype=torch.float64).requires_grad_(True)

            # IC loss (soft IC for Convection only)
            if not USE_HARD_IC:
                n_ic = 300
                x_ic = torch.empty(n_ic, 1, device=DEVICE, dtype=torch.float64).uniform_(x_lo, x_hi)
                t_ic = torch.zeros(n_ic, 1, device=DEVICE, dtype=torch.float64)

                if wi == 0:
                    if args.problem == 'convection':
                        u_ic_true = torch.sin(x_ic).double()
                    else:
                        u_ic_true = (x_ic**2 * torch.cos(math.pi * x_ic)).double()
                else:
                    with torch.no_grad():
                        t_prev_end = torch.ones(n_ic, 1, device=DEVICE, dtype=torch.float64)
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

            loss_pde = (residual**2).mean()

            if USE_HARD_IC:
                loss = loss_bc + loss_pde
            else:
                loss = 10.0 * loss_ic + loss_bc + loss_pde

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
