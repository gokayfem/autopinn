# AutoPINN

> You are a completely autonomous AI researcher. Your job is to improve
> Physics-Informed Neural Networks (PINNs) by editing `pinn.py`, running
> experiments, keeping improvements, discarding failures, and advancing
> indefinitely. You do the science. You decide what to try next.
>
> **The only file you edit is `pinn.py`.**

---

## Setup (do once at the start of each session)

**Read the codebase in full before your first experiment:**
- `program.md` — this file (read it all)
- `pinn.py` — the file you modify: architecture, optimizer, training loop
- `problems/allen_cahn.py` — PDE definition, sampling, evaluation. Do not modify.
- `problems/convection.py` — PDE definition, sampling, evaluation. Do not modify.

**Initialize the log:**  
Create `results.tsv` with the header and baseline rows below. Do NOT re-run
the baseline — the numbers are already known.

```
commit	problem	L2_error	status	description
baseline	Convection	1.000000	keep	vanilla MLP 4×64 tanh, Adam lr=1e-3
baseline	Allen-Cahn	0.500000	keep	vanilla MLP 4×64 tanh, Adam lr=1e-3
```

**Confirm setup and begin.** Once you confirm setup looks good, kick off
experiments immediately. Do not wait for further permission.

---

## The Experiment Loop

Each experiment runs on the local GPU with a fixed 3-minute
wall-clock budget.

```bash
python pinn.py --problem convection   > run.log 2>&1   # fast idea validation
python pinn.py --problem allen_cahn   > run.log 2>&1   # primary target
```

**Read the result:**
```bash
grep "^RESULT" run.log
```

If grep returns nothing, the run crashed. Run `tail -50 run.log` for the
traceback. If it's a trivial bug (typo, missing import), fix and re-run.
If the idea is fundamentally broken, mark as `crash` in the log and move on.

**After each run:**
1. Record the result in `results.tsv` (tab-separated).
2. If L2 improved: keep the change, commit it, update Active State below.
3. If L2 did not improve: revert `pinn.py` to the last kept state.
4. Form the next hypothesis from what you just learned.

**TSV format:**
```
a1b2c3d	Convection	0.847100	keep	Fourier features sigma=10, n=256
b2c3d4e	Convection	0.901200	discard	increased depth to 8 layers (overfit)
c3d4e5f	Allen-Cahn	0.001510	keep	causal loss eps=1.0 + Fourier features
d4e5f6g	Allen-Cahn	0.000000	crash	PirateNet: OOM at width=512
```

**NEVER STOP.** Once the loop begins, do not ask whether to continue. The
human may be asleep. Keep experimenting until you are manually interrupted.
If you run out of ideas, re-read the failure modes section, re-examine
what near-misses share in common, try combining two previously-partial
improvements, or try something more radical. The loop runs indefinitely.

---

## Active State

*(You update this section when you keep a change — this is your memory of
what is currently in pinn.py)*

```
Best scores:
  Convection : 1.000  (baseline — not started)
  Allen-Cahn : 0.500  (baseline — not started)

Current pinn.py state:
  Architecture : vanilla MLP  [4 × 64, tanh]
  Optimizer    : Adam  lr=1e-3
  Loss weights : uniform  (w_pde=1, w_ic=1, w_bc=1)
  Collocation  : uniform random  5000 interior pts
  Precision    : float32
  Other        : none
```

---

## The PDEs

### Convection — use this first to validate ideas quickly

```
u_t + 50·u_x = 0,   x ∈ [0, 2π],   t ∈ [0, 1]
IC:  u(x, 0) = sin(x)
BC:  periodic
Exact: u(x, t) = sin(x − 50t)
```

Metric: L2 relative error vs exact solution. Lower is better.  
Baseline: ~1.000. Milestones: <0.10 meaningful | <1e-03 good | <1e-04 SOTA.

### Allen-Cahn — primary research target

```
u_t − ε²·u_xx + 5·(u³ − u) = 0,   x ∈ [−1, 1],   t ∈ [0, 1]
ε = 0.001
IC:  u(x, 0) = x²·cos(πx)
BC:  periodic  (u(−1, t) = u(1, t),  u_x(−1, t) = u_x(1, t))
```

Metric: L2 relative error vs spectral reference (512 Fourier modes, RK45
with rtol=1e-10). This is the identical reference used by Raissi 2019,
Wang 2022, PirateNet JMLR 2024 — directly comparable to the literature.  
Baseline: ~0.50. Milestones: <1e-02 progress | <1e-03 matches Causal PINN | <1e-04 excellent.

---

## SOTA Ladder

These are verified numbers from peer-reviewed papers. Your goal is to climb
this ladder. The numbers tell you what's achievable — figuring out *how*
to achieve it is your job.

**Convection:**
| L2 Error | Method | Source |
|----------|--------|--------|
| ~1.000 | Vanilla PINN | Krishnapriyan NeurIPS 2021 |
| ~0.050 | Curriculum learning | Krishnapriyan NeurIPS 2021 |
| ~1e-03 | Causal PINN | Wang CMAME 2024 |
| ~1e-04 | FP64 vanilla PINN | arXiv 2025 |
| ~1e-04 | ProPINN | Wu et al. arXiv 2502.00803 |

**Allen-Cahn:**
| L2 Error | Method | Source |
|----------|--------|--------|
| ~0.500 | Vanilla PINN | Wang CMAME 2022 |
| 1.51e-03 | Causal PINN | IJCAI 2024 |
| 5.37e-05 | JAX-PI | Wang benchmark 2024 |
| 2.24e-05 | PirateNet | JMLR 2024 |
| 3.26e-06 | Deeper-PINNs (best ablation) | ResearchGate 2024 |

---

## Why the Baseline Fails — Physics and Math of the Failures

**Read this section carefully before your first experiment.** The baseline
scores L2≈1.0 on Convection and L2≈0.5 on Allen-Cahn. These are not slow
convergence — they are complete failure. You need to understand *why* in
order to form hypotheses worth testing.

### Failure 1: Spectral Bias

Neural networks trained with gradient descent learn low spatial frequencies
first and high frequencies extremely slowly (Rahaman et al. 2019, NeurIPS).
This is a fundamental property of the NTK (neural tangent kernel) eigenvalue
spectrum: the largest eigenvalues correspond to smooth, low-frequency
eigenfunctions, so gradients point overwhelmingly toward fitting those first.

The Allen-Cahn equation has a sharp interface of characteristic width ε=0.001
— a near-discontinuity in u(x,t) at the phase boundary. A 4-layer tanh MLP
will never represent this structure in 3 minutes; the gradients needed to fit
the high-frequency content are ~1000× smaller than gradients for the smooth
background.

The Convection problem at β=50 requires the network to represent sin(x−50t):
a function that oscillates 50 times faster in time than in space. This extreme
anisotropy is exactly the regime where spectral bias causes complete failure.

*What this suggests:* The input representation matters enormously. If you
project inputs into a feature space that already contains the required
frequencies, gradient descent can fit them. This is the motivation behind
random Fourier features, learned frequency embeddings, and periodic activations.

### Failure 2: Causality Violation

The PDE is an initial-value problem: the solution at time t can only be
determined from the solution at times t'<t. But the PINN loss minimizes
all collocation points simultaneously, with no causal ordering:

```
L = Σᵢ r(xᵢ, tᵢ)²
```

This means the optimizer can satisfy the PDE residual at t=0.9 while getting
t=0.1 completely wrong — and it does, because fitting t=0.9 might be
"easier" (lower gradient norm) in the early optimization landscape. Once the
network finds a spurious solution at late times, it is locally optimal and
gradient descent won't escape.

For Convection with β=50, there are approximately 50 distinct failure modes
(one per wave cycle) where the network fits a shifted version of the true
solution. Each is a local minimum of the unconstrained PDE loss. This is
why curriculum learning (fitting t∈[0,0.1] first, then extending) and causal
loss weighting (penalizing residuals at t=T only after residuals at t<T are
satisfied) both help substantially.

*What this suggests:* Some form of temporal ordering — either in sampling
(biasing toward early t), in loss weighting (downweight late t until early
t is solved), or in domain decomposition (solve sequential time windows) —
should make a large difference. The exact best form is your research question.

### Failure 3: Loss Imbalance and Gradient Pathology

The total PINN loss is:
```
L = w_pde·L_pde + w_ic·L_ic + w_bc·L_bc
```

In early training with uniform weights (w=1), the IC loss starts large
and dominates. The optimizer fits the initial condition at the expense of
the PDE residual. But once L_ic is small, L_pde gradients compete against
near-zero IC gradients, and updates simultaneously push in conflicting
directions across these three objectives.

The NTK analysis of PINNs (Wang et al. 2021, SIAM J. Sci. Comput.) shows
that under standard initialization, the eigenvalue spectra of the three NTK
sub-matrices (for PDE, IC, BC) are severely mismatched — the IC NTK has
much larger eigenvalues, meaning the network trains much faster on IC than
on the PDE. This is the precise mathematical cause of loss imbalance.

*What this suggests:* Loss weights that adapt during training — either by
monitoring gradient norms and normalizing, or by learning separate scale
parameters — should address this. The fixed ratio (1,1,1) is known to be
suboptimal; the optimal ratio likely evolves as training progresses.

### Failure 4: Precision Artifacts in Optimization

L-BFGS in float32 approximates the Hessian using float32 dot products. When
the gradient norm falls to ~1e-7 (which happens in the flat region of PINN
loss landscapes near local minima), float32 arithmetic introduces errors of
~1e-8, close to machine epsilon (1.2e-7). The BFGS convergence criterion is
satisfied not because the true minimum was reached, but because the gradient
appears small within float32's representable range.

In float64, machine epsilon is ~2.2e-16, so the same gradient of 1e-7 is
clearly non-zero and L-BFGS continues descending. A 2025 paper showed this
single change (float32 → float64 + L-BFGS) was sufficient to solve the
Convection benchmark at β=50 with a vanilla PINN that always failed in float32.

*What this suggests:* `.double()` on the model and float64 tensor casting
is a nearly free change that might give a large improvement, particularly
paired with L-BFGS. The tradeoff is 2× memory and ~2× compute per step,
meaning fewer steps in the 3-minute budget.

### Failure 5: Architecture Mismatch

The vanilla MLP treats x and t as symmetric inputs, but they play
fundamentally different roles. The PDE is first-order in t and second-order
in x; the IC imposes a known function at t=0; the solution structure
(sharp interface moving in time) is highly anisotropic.

Standard Xavier/Kaiming initialization produces outputs near zero everywhere
at initialization. For Allen-Cahn where |u|≈1 almost everywhere after a
short transient, this is a poor starting point — the network must discover
the ±1 phase structure purely from gradient descent, which is slow.

Hard enforcement of the IC via output transform:
```
u_pred = IC(x) + t · NN(x, t)
```
forces u(x,0)=IC(x) exactly with zero IC loss from step 1, freeing all
gradient capacity for the PDE residual. This is not a technique from the
search space — it is a structural change that removes one failure mode
entirely.

*What this suggests:* Architecture choices that encode known structure
can be more powerful than loss weight tuning. The IC is known exactly.
Using it as a hard constraint rather than a soft loss term costs nothing
and removes one of the three loss terms entirely.

---

## Simplicity Criterion

All else being equal, simpler is better.

A small improvement that adds 20 lines of hacky code is probably not worth it.
An improvement of ~0 but much simpler code? Keep it — that is a win.
Removing something and getting equal or better results? That is the best outcome.

When deciding whether to keep a change, weigh the complexity cost against the
improvement magnitude. The cleaner `pinn.py` is, the more room there is to
build on it. A messy pinn.py after 50 experiments is a research dead end.

---

## Practical Notes

- **Test ideas on Convection first.** It runs the same failure modes, has
  an exact solution (no reference precomputation), and gives clean gradient
  signal faster. Once an idea shows improvement on Convection, test Allen-Cahn.

- **One change at a time.** If you combine 3 ideas at once and it works,
  you don't know which one helped. Combine only after individual effects
  are established.

- **Do not change `--seed` or `BUDGET_SECONDS`** to get better numbers.
  Seed changes are not improvements. The budget must stay at 180s so all
  experiments are comparable. Never modify the `evaluate()` call.

- **The problems/ directory is fixed.** `pde_residual()`, `evaluate()`,
  `sample_interior()`, `sample_initial()`, `sample_periodic_bc()` have
  fixed signatures. You call them exactly as the baseline does.

- **Commits mark kept experiments.** Use `git add pinn.py && git commit -m
  "exp-N: <description>"` after every kept improvement.

- **VRAM:** If you go OOM, log as `crash` and reduce model size or batch size.

---

## Session Report (write at end of each session)

When you finish a session (or are interrupted), write a session report to
`sessions/session-YYYYMMDD-HHMM.md` with this structure:

```markdown
# Session Report — <date>

Starting best: Convection=X.XXX  Allen-Cahn=X.XXX
Ending best:   Convection=X.XXX  Allen-Cahn=X.XXX
Experiments run: N  |  Kept: K

## Top wins
| Delta | Description |
|-------|-------------|
| -0.XXX | ... |

## What did NOT work
- X: tried Y, got +0.XXX (worse) — reason if known

## Fragile / seed-sensitive results
- List anything that improved but may not reproduce

## Final pinn.py state
Architecture: ...
Optimizer: ...
Key changes from baseline: ...

## Suggested next experiments
- ...
```

At the **start** of each session, read all files in `sessions/` before
your first experiment. Do not repeat things that previous sessions tried
and found did not work.

---

## Experiment Log

*(You fill this in as experiments run)*

| # | Problem | L2 | Status | Description |
|---|---------|-----|--------|-------------|
| 0 | Convection | 1.000 | baseline | vanilla MLP 4×64 tanh, Adam |
| 0 | Allen-Cahn | 0.500 | baseline | vanilla MLP 4×64 tanh, Adam |
