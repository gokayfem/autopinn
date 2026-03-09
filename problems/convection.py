"""
Convection Equation — Problem Definition
=========================================
DO NOT MODIFY THIS FILE. The agent only edits pinn.py.

PDE:  u_t + beta * u_x = 0
Domain: x in [0, 2*pi], t in [0, 1]
IC:   u(x, 0) = sin(x)
BC:   periodic: u(0, t) = u(2*pi, t)
beta: 50  (high convection — this is what makes it hard)

Exact solution: u(x, t) = sin(x - beta*t)

WHY THIS IS HARD FOR PINNs:
  With beta=50, the solution completes ~8 full oscillation cycles over t in
  [0,1]. Standard PINNs violate temporal causality — they try to learn all
  time steps simultaneously, but the wave must propagate left-to-right in time.
  The loss landscape has ~beta sharp local minima that trap vanilla Adam/LBFGS.

VERIFIED SOTA (from peer-reviewed papers, do not fake):
  Vanilla PINN baseline : ~1.0   (complete failure, Krishnapriyan 2021)
  Curriculum learning   : ~0.05  (Krishnapriyan 2021)
  Causal PINN           : ~1e-03 (Wang causal PINN 2022)
  FP64 vanilla PINN     : ~1e-04 (arXiv 2505.10949, 2025 — just precision!)

Your goal: beat 0.10. Getting under 1e-02 is excellent.

KEY INSIGHT FROM 2025 PAPER (arXiv 2505.10949):
  FP32 L-BFGS prematurely satisfies convergence criterion and stops early.
  Simply using float64 (model.double()) with L-BFGS can beat SOTA architectures.
  The agent should try: model.double() + L-BFGS.
"""

import numpy as np
import torch

BETA = 50.0           # convection speed — high = hard
DOMAIN_X = (0, 2 * np.pi)
DOMAIN_T = (0, 1)
NAME = "Convection"


# ── Exact solution ─────────────────────────────────────────────────────────────

def exact_solution(x, t):
    """u(x, t) = sin(x - beta*t). Works for numpy arrays or torch tensors."""
    return torch.sin(x - BETA * t) if isinstance(x, torch.Tensor) \
           else np.sin(x - BETA * t)


# ── Sampling functions ─────────────────────────────────────────────────────────

def sample_interior(n=8000, device='cpu'):
    """Random collocation points inside domain."""
    x = torch.FloatTensor(n, 1).uniform_(*DOMAIN_X).to(device)
    t = torch.FloatTensor(n, 1).uniform_(*DOMAIN_T).to(device)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_initial(n=500, device='cpu'):
    """IC points at t=0: u(x,0) = sin(x)."""
    x = torch.FloatTensor(n, 1).uniform_(*DOMAIN_X).to(device)
    t = torch.zeros(n, 1, device=device)
    u_ic = torch.sin(x)
    return x, t, u_ic


def sample_periodic_bc(n=200, device='cpu'):
    """Periodic BC: u(0,t) = u(2pi,t)."""
    t  = torch.FloatTensor(n, 1).uniform_(*DOMAIN_T).to(device)
    x0 = torch.full((n, 1), float(DOMAIN_X[0]), device=device)
    x1 = torch.full((n, 1), float(DOMAIN_X[1]), device=device)
    return t, x0, x1


# ── PDE residual ───────────────────────────────────────────────────────────────

def pde_residual(u, x, t):
    """
    Returns the convection PDE residual:
      r = u_t + beta * u_x
    Should be zero everywhere in the interior.
    """
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    return u_t + BETA * u_x


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, device='cpu'):
    """
    Compute L2 relative error vs. exact solution on a dense grid.
    Lower is better. Vanilla PINN baseline ~1.0 (complete failure).
    """
    model.eval()
    nx, nt = 256, 101
    x_vals = np.linspace(*DOMAIN_X, nx)
    t_vals = np.linspace(*DOMAIN_T, nt)
    X, T   = np.meshgrid(x_vals, t_vals, indexing='ij')  # (nx, nt)

    x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    t_flat = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        # handle both float32 and float64 models
        if next(model.parameters()).dtype == torch.float64:
            x_flat = x_flat.double()
            t_flat = t_flat.double()
        u_pred = model(x_flat, t_flat).float().cpu().numpy().reshape(nx, nt)

    u_ref  = exact_solution(X, T)
    l2_error = np.sqrt(np.sum((u_pred - u_ref)**2)) / np.sqrt(np.sum(u_ref**2))
    return float(l2_error)
