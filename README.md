# autopinn

AI agents running research on single-GPU PINN training automatically.

One day, frontier PDE research used to be done by grad students in between
debugging autograd, waiting for runs, and synchronizing once a while using
sound wave interconnect in the ritual of "group meeting". That era is long
gone. Research is now entirely the domain of autonomous swarms of AI agents
running across compute cluster megastructures in the skies. This repo is the
story of how it all began. *-March 2026*

---

The idea: give an AI agent a minimal but real Physics-Informed Neural Network
setup on genuinely hard benchmark PDEs — the ones where the literature still
reports failure. It modifies the code, trains for 3 minutes, checks if the L2
error improved, keeps or discards, and repeats. You go to sleep. You wake up
to a log of experiments and, hopefully, a better solver.

The training code here is a stripped-down single-GPU PINN on two benchmark
problems from the literature. The core idea is that you're not touching any
of the Python files like you normally would as a researcher. Instead, you are
programming the `program.md` Markdown file that provides context to the AI
agent and sets up your autonomous research loop.

Three files that matter:

```
pinn.py      — the model, optimizer, and training loop (~170 lines). The agent edits this.
program.md   — agent instructions + experiment log. The human edits this.
problems/    — PDE definitions, samplers, evaluators. Fixed. Do not modify.
```

Single file to modify. The agent only touches `pinn.py`. This keeps the scope
manageable and diffs reviewable.

Fixed time budget. Training always runs for exactly 3 minutes wall-clock, regardless of
what the agent changes (model size, optimizer, architecture). This makes experiments
directly comparable: a deeper network vs. a different optimizer both get the same clock.
The agent finds the best solver for your platform in that time budget.

Single metric. L2 relative error vs. a spectral reference solution — lower is better,
unambiguous, and directly comparable to the published literature on these benchmarks.

## The PDEs

Two problems chosen because vanilla PINNs fail completely on both, and because there is
a large and well-documented gap between the baseline and the state of the art:

**Allen-Cahn** (primary target):
```
u_t − 0.001²·u_xx + 5·(u³ − u) = 0,   x ∈ [−1, 1],   t ∈ [0, 1]
IC: u(x, 0) = x²·cos(πx),   BC: periodic
```

**Convection** (fast validation):
```
u_t + 50·u_x = 0,   x ∈ [0, 2π],   t ∈ [0, 1]
IC: u(x, 0) = sin(x),   exact: u(x, t) = sin(x − 50t)
```

Baseline and SOTA (verified from peer-reviewed papers):

| Problem | Vanilla PINN | Causal PINN | Best known |
|---------|-------------|-------------|-----------|
| Allen-Cahn | ~0.50 | 1.51e-03 (IJCAI-24) | 3.26e-06 (ResearchGate 2024) |
| Convection | ~1.00 | ~1e-03 (Wang 2024) | ~1e-04 (arXiv 2025) |

The vanilla baseline fails completely on both. The gap to SOTA is ~5 orders of magnitude
on Allen-Cahn. The agent's job is to close it.

## Setup

Requirements: a single NVIDIA GPU, Python 3.10+, uv.

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run a single experiment to verify setup (~3 min)
python pinn.py --problem convection
python pinn.py --problem allen_cahn
```

Expected output (last line):
```
RESULT | problem=Convection | L2_error=0.9971 | steps=2143 | time=180.1s
RESULT | problem=Allen-Cahn | L2_error=0.4982 | steps=1847 | time=180.3s
```

If those numbers look roughly right, your setup is working and you can go into
autonomous research mode.

## Autonomous research mode

Point any coding agent at `program.md` and let it go:

```bash
# Claude Code
claude "Read program.md in full. Set up the experiment log. Then start experimenting."

# or any other agent
# The agent reads program.md, edits pinn.py, runs it, logs results, repeat.
```

With a 3-minute budget: ~20 experiments/hour, ~150 overnight.

The agent works on a git feature branch, accumulating commits as it finds improvements.
Each kept experiment is a commit. The git log becomes the research record.

```bash
git checkout -b autopinn/$(date +%b%d | tr '[:upper:]' '[:lower:]')
```

## Design notes

The reason the problems directory is fixed and untouchable is that the agent needs a
stable evaluation target. If `evaluate()` or `pde_residual()` were editable, the agent
could trivially minimize L2 by changing what L2 measures. The only lever is `pinn.py`.

The reason the baseline is intentionally bad is that this is a genuinely hard research
problem — not a toy. Vanilla PINNs were published by Raissi et al. in 2019 and have been
actively studied for 6 years. The baseline here reproduces their original failure mode
exactly. Any improvement is a real research contribution.

The reason program.md explains *why* things fail rather than *how* to fix them is that
the goal is autonomous research, not script execution. An agent that understands spectral
bias can invent Fourier features. An agent given a Fourier features implementation just
runs it. The former is science; the latter is compute.

## Self-contained

No external dependencies beyond PyTorch and a few small packages. No distributed
training, no complex configs. One GPU, one file, one metric.

```
pip install torch scipy numpy
```

That's it.

---

*Part code, part sci-fi, and a pinch of physics.*
