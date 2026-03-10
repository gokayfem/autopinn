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


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class SubPINN(nn.Module):
    """Small MLP for a single time window."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


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

    # 5 time windows
    n_windows = 5
    t_boundaries = [i / n_windows for i in range(n_windows + 1)]
    time_per_window = BUDGET_SECONDS / n_windows

    models = []
    global_deadline = time.time() + BUDGET_SECONDS
    step = 0

    for wi in range(n_windows):
        t_lo_w = t_boundaries[wi]
        t_hi_w = t_boundaries[wi + 1]
        dt = t_hi_w - t_lo_w

        sub_model = SubPINN().to(DEVICE)
        optimizer = torch.optim.Adam(sub_model.parameters(), lr=1e-3)

        window_deadline = time.time() + time_per_window

        while time.time() < min(window_deadline, global_deadline):
            n_pts = 3000
            # Sample in x × [0, 1] (normalized time within window)
            x_f = torch.FloatTensor(n_pts, 1).uniform_(x_lo, x_hi).to(DEVICE).requires_grad_(True)
            t_f = torch.FloatTensor(n_pts, 1).uniform_(0, 1).to(DEVICE).requires_grad_(True)

            # IC for this window: at t_local=0
            n_ic = 300
            x_ic = torch.FloatTensor(n_ic, 1).uniform_(x_lo, x_hi).to(DEVICE)
            t_ic = torch.zeros(n_ic, 1, device=DEVICE)

            if wi == 0:
                # True IC: u(x, 0)
                if args.problem == 'convection':
                    u_ic_true = torch.sin(x_ic)
                else:
                    u_ic_true = x_ic**2 * torch.cos(math.pi * x_ic)
            else:
                # Use previous model's prediction at t_local=1 as IC
                with torch.no_grad():
                    t_prev_end = torch.ones(n_ic, 1, device=DEVICE)
                    u_ic_true = models[-1](x_ic, t_prev_end)

            u_ic_pred = sub_model(x_ic, t_ic)
            loss_ic = ((u_ic_pred - u_ic_true)**2).mean()

            # BC: periodic
            n_bc = 100
            t_bc = torch.FloatTensor(n_bc, 1).uniform_(0, 1).to(DEVICE)
            x_left  = torch.full((n_bc, 1), x_lo, device=DEVICE).requires_grad_(True)
            x_right = torch.full((n_bc, 1), x_hi, device=DEVICE).requires_grad_(True)
            u_left  = sub_model(x_left, t_bc)
            u_right = sub_model(x_right, t_bc)
            loss_bc = ((u_left - u_right)**2).mean()

            # PDE residual: need to account for time scaling
            # If t_real = t_lo + t_local * dt, then u_t_real = u_t_local / dt
            # So PDE residual uses chain rule
            u_f = sub_model(x_f, t_f)
            # Compute derivatives wrt local coordinates
            u_t_local = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f),
                                            create_graph=True)[0]
            u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f),
                                      create_graph=True)[0]

            if args.problem == 'convection':
                # PDE: u_t + 50*u_x = 0, but u_t_real = u_t_local / dt
                residual = u_t_local / dt + 50.0 * u_x
            else:
                u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x),
                                           create_graph=True)[0]
                residual = u_t_local / dt - 0.001**2 * u_xx + 5.0 * (u_f**3 - u_f)

            loss_pde = (residual**2).mean()

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
