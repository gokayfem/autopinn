# Security boundaries

This project trains its own numerical network from source. It does not accept model files or checkpoints, and the current code has no `torch.load`, `torch.jit.load`, `torch.export.load`, or `.pt2` deserialization path. Do not run untrusted Python experiments or introduce untrusted model-loading features. Any future checkpoint import needs its own security review and a safe, restricted format.

The CUDA 12.8 dependency lane remains explicit. The fresh dependency audit still reports two PyTorch 2.10.0 advisories: CVE-2025-3000 (fixed upstream in 2.13.0) and CVE-2026-4538 (no fix listed). The current source does not expose the reported JIT/PT2-loading entry points, but this does not make the installed library vulnerability-free. The CUDA 12.8 package index currently has no Torch 2.13 wheel; upgrading to another CUDA lane requires checking the target GPU and driver first.

Pillow, filelock, fsspec, and setuptools are constrained to patched releases. Use Python 3.11+ and the checked-in lockfile. Published historical performance measurements are not a compatibility or security claim for a new dependency set.
