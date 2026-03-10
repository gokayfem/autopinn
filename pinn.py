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

WHAT TO TRY (read program.md for the full research context):
  - Fourier feature input embedding  → fights spectral bias
  - Causal/time-weighted loss        → respects causality (biggest win)
  - Self-adaptive loss weights       → learnable lambda per loss term
  - Adaptive collocation resampling  → focus points on high residual regions
  - L-BFGS fine-tune after Adam      → second-order convergence
  - float64 precision                → 2025 paper shows this alone can beat SOTA
  - Deeper residual architecture     → PirateNet shows deep nets can work
  - Hard BC enforcement via transform → remove BC loss term entirely
  - Domain/time decomposition        → break [0,1] into sequential windows
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
# ARCHITECTURE — agent edits this
# ══════════════════════════════════════════════════════════════════════════════

class PINN(nn.Module):
    """
    Vanilla MLP — intentionally basic baseline.
    The agent should improve this architecture.

    Known weaknesses:
      - tanh has spectral bias: learns low frequencies first, ignores sharp
        interface in Allen-Cahn
      - No positional encoding: misses high-frequency structure
      - Uniform depth: no adaptive capacity allocation
      - Standard init: poor gradient flow for deep PINNs
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — agent edits this
# ══════════════════════════════════════════════════════════════════════════════

def train():
    model = PINN().double().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    w_pde = 1.0
    w_ic  = 1.0
    w_bc  = 1.0

    deadline = time.time() + BUDGET_SECONDS
    step = 0

    while time.time() < deadline:

        x_f, t_f = sample_interior(n=5000, device=DEVICE)
        x_f = x_f.double()
        t_f = t_f.double()

        x_ic, t_ic, u_ic_true = sample_initial(n=500, device=DEVICE)
        x_ic  = x_ic.double().requires_grad_(False)
        t_ic  = t_ic.double().requires_grad_(False)
        u_ic_true = u_ic_true.double()
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = ((u_ic_pred - u_ic_true)**2).mean()

        t_bc, x_left, x_right = sample_periodic_bc(n=200, device=DEVICE)
        t_bc = t_bc.double()
        x_left  = x_left.double().requires_grad_(True)
        x_right = x_right.double().requires_grad_(True)
        u_left  = model(x_left, t_bc)
        u_right = model(x_right, t_bc)
        loss_bc = ((u_left - u_right)**2).mean()

        u_f   = model(x_f, t_f)
        res   = pde_residual(u_f, x_f, t_f)
        loss_pde = (res**2).mean()

        loss = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

    # ── Evaluate ───────────────────────────────────────────────────────────────
    elapsed = BUDGET_SECONDS  # approximately
    l2 = evaluate(model, DEVICE)

    print(f"RESULT | run={args.run_id} | seed={args.seed} | problem={NAME} | "
          f"L2_error={l2:.6f} | steps={step} | time={elapsed:.1f}s")
    return l2


if __name__ == '__main__':
    t0 = time.time()
    result = train()
    print(f"Total wall time: {time.time()-t0:.1f}s")
