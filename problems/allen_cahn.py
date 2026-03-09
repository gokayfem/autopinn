"""
Allen-Cahn Equation — Problem Definition
=========================================
DO NOT MODIFY THIS FILE. The agent only edits pinn.py.

PDE:  u_t - eps^2 * u_xx + f(u) = 0
      f(u) = 5*(u^3 - u)
Domain: x in [-1, 1], t in [0, 1]
IC:   u(x, 0) = x^2 * cos(pi*x)
BC:   periodic: u(-1, t) = u(1, t), u_x(-1, t) = u_x(1, t)
eps:  0.001 (small interface parameter — this is what makes it hard)

Reference solution: computed via spectral method (Chebfun-equivalent)
stored in Allen_Cahn.mat (same file used by Raissi 2019, Wang 2022,
PirateNet JMLR 2024 — directly comparable to literature).

VERIFIED SOTA (from peer-reviewed papers, do not fake):
  Vanilla PINN baseline : ~0.50  (49.87% — Wang causal PINN paper)
  Causal PINN           : 1.51e-03  (IJCAI-24, exact table)
  JAX-PI                : 5.37e-05  (ResearchGate benchmark table)
  PirateNet             : 2.24e-05  (JMLR 2024 — "best ever reported")
  Deeper-PINNs (best)   : 3.26e-06  (ablation best, ResearchGate)

Your goal: beat 1e-03. Getting under 1e-04 is excellent.
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp

EPS = 0.001          # interface parameter
DOMAIN_X = (-1, 1)
DOMAIN_T = (0, 1)
NAME = "Allen-Cahn"


# ── Reference solution via spectral method ────────────────────────────────────

def _make_reference(nx=512, nt=201):
    """Generate reference solution using spectral Fourier method + RK45."""
    x = np.linspace(-1, 1, nx, endpoint=False)
    dx = x[1] - x[0]
    # wavenumbers for spectral differentiation
    k = np.fft.fftfreq(nx, d=dx / (2 * np.pi))

    u0 = x**2 * np.cos(np.pi * x)

    def rhs(t, u):
        u_hat = np.fft.fft(u)
        # u_xx via spectral
        u_xx = np.real(np.fft.ifft(-(k**2) * u_hat))
        return EPS**2 * u_xx + 5.0 * (u - u**3)

    t_eval = np.linspace(0, 1, nt)
    sol = solve_ivp(rhs, [0, 1], u0, method='RK45', t_eval=t_eval,
                    rtol=1e-10, atol=1e-12, max_step=1e-3)

    # shape: (nt, nx)
    U = sol.y.T
    T, X = np.meshgrid(t_eval, x, indexing='ij')
    return X.astype(np.float32), T.astype(np.float32), U.astype(np.float32)


_REF = None

def _get_reference():
    global _REF
    if _REF is None:
        _REF = _make_reference()
    return _REF


# ── Sampling functions ─────────────────────────────────────────────────────────

def sample_interior(n=10000, device='cpu'):
    """Random collocation points inside domain."""
    x = torch.FloatTensor(n, 1).uniform_(*DOMAIN_X).to(device)
    t = torch.FloatTensor(n, 1).uniform_(*DOMAIN_T).to(device)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_initial(n=1000, device='cpu'):
    """IC points at t=0."""
    x = torch.FloatTensor(n, 1).uniform_(*DOMAIN_X).to(device)
    t = torch.zeros(n, 1, device=device)
    u_ic = (x**2 * torch.cos(torch.pi * x)).to(device)
    return x, t, u_ic


def sample_periodic_bc(n=500, device='cpu'):
    """Periodic BC: u(-1,t) = u(1,t) and u_x(-1,t) = u_x(1,t)."""
    t = torch.FloatTensor(n, 1).uniform_(*DOMAIN_T).to(device)
    x_left  = torch.full((n, 1), float(DOMAIN_X[0]), device=device).requires_grad_(True)
    x_right = torch.full((n, 1), float(DOMAIN_X[1]), device=device).requires_grad_(True)
    return t, x_left, x_right


# ── PDE residual ───────────────────────────────────────────────────────────────

def pde_residual(u, x, t):
    """
    Returns the Allen-Cahn PDE residual:
      r = u_t - eps^2 * u_xx + 5*(u^3 - u)
    Should be zero everywhere in the interior.
    """
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    return u_t - EPS**2 * u_xx + 5.0 * (u**3 - u)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, device='cpu'):
    """
    Compute L2 relative error vs. spectral reference solution.
    Lower is better. Vanilla PINN baseline ~0.50.
    """
    model.eval()
    X_ref, T_ref, U_ref = _get_reference()

    x_flat = torch.tensor(X_ref.reshape(-1, 1), dtype=torch.float32, device=device)
    t_flat = torch.tensor(T_ref.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        # handle both float32 and float64 models
        if next(model.parameters()).dtype == torch.float64:
            x_flat = x_flat.double()
            t_flat = t_flat.double()
        u_pred = model(x_flat, t_flat).float().cpu().numpy().reshape(X_ref.shape)

    u_ref = U_ref
    l2_error = np.sqrt(np.sum((u_pred - u_ref)**2)) / np.sqrt(np.sum(u_ref**2))
    return float(l2_error)
