"""Function-preserving rotated-basis LoRA via torch.nn.utils.parametrize.

Matches the prior implementation at
`all_experiments/train/rotated_residual.py` on the clean branch:

- Targets are the **MLP side only**: each transformer layer's
  `{up_proj, gate_proj}.lora_A` (read-side) and `{down_proj}.lora_B`
  (write-side). All three weights interact with the 3584-dim residual
  stream on the rotated axis, so a single 3584×3584 R matrix sizes
  every parametrization. Attention LoRA is NOT rotated (q/o_proj have
  hidden_dim=3584 but k/v_proj have out_features=512 due to GQA, and
  rotating that breaks the GQA grouping; the prior implementation
  also skipped attention).
- Read-side rotation:  `A_eff = A @ R.T`  (weight shape (r, 3584))
- Write-side rotation: `B_eff = R @ B`    (weight shape (3584, r))

Because LoRA initialises `B = 0`, `B_eff = R @ 0 = 0` regardless of R,
so the model's forward function at init is identical to the standard
basis. During training the optimiser sees a rotated parameterisation;
trajectories differ for adaptive optimisers (Adam, RMSProp) because
their per-coord variance estimate tracks the rotated gradient. Plain
SGD is basis-equivariant, so SGD-rotated == SGD-standard exactly
(this is why the experiment is only informative under AdamW).

Two modes:

- `per_layer=False` (shared R): one 3584×3584 R reused across all 84
  parametrized modules. Cheap (~51 MB on GPU).
- `per_layer=True`: one fresh R per transformer layer, seeded as
  (seed + layer_idx + 1). 28 R matrices × 51 MB = ~1.4 GB on GPU.

Call `apply_rotated_basis(model, seed, per_layer)` AFTER the PEFT
adapter is attached and before training; call
`bake_parametrizations(model)` once before `trainer.save_model()` so
the saved adapter is a plain LoRA adapter that needs no parametrization
at eval time.
"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

_READ_PARENTS = {"up_proj", "gate_proj"}
_WRITE_PARENTS = {"down_proj"}


def make_orthogonal_matrix(dim: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """QR of a Gaussian (dim, dim) with sign-flip so det(Q) = +1. Runs on GPU
    by default — CPU QR on 3584×3584 is fast (~1s) but on 18944 it's minutes;
    we keep `device` configurable so callers can force CPU for smaller dims."""
    rng = torch.Generator(device=device).manual_seed(seed)
    g = torch.randn(dim, dim, generator=rng, dtype=torch.float32, device=device)
    Q, R = torch.linalg.qr(g)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    err = (Q.T @ Q - torch.eye(dim, device=device)).norm().item()
    assert err < 1e-2, f"||R^T R - I||_F = {err:.4g}, expected < 1e-2"
    return Q


class RotateLoraB(nn.Module):
    """Write-side parametrization: B_eff = R @ B. weight shape (3584, r)."""

    def __init__(self, R: torch.Tensor):
        super().__init__()
        self.register_buffer("R", R)

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        return self.R.to(device=B.device, dtype=B.dtype) @ B


class RotateLoraA(nn.Module):
    """Read-side parametrization: A_eff = A @ R.T. weight shape (r, 3584)."""

    def __init__(self, R: torch.Tensor):
        super().__init__()
        self.register_buffer("R", R)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        return A @ self.R.to(device=A.device, dtype=A.dtype).T


def _parent_proj(name: str) -> str | None:
    """Extract the projection name (e.g. 'down_proj') from a PEFT module path
    'base_model.model.model.layers.13.mlp.down_proj.lora_B.default'."""
    parts = name.split(".")
    # PEFT names end in '<proj>.lora_<A|B>.default'
    if len(parts) < 3:
        return None
    return parts[-3]


def _layer_idx(name: str) -> int:
    """Pick the first integer-looking dotted part as the layer index."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return 0


def _infer_hidden_dim(model) -> int:
    """Locate `model.model.layers[0].input_layernorm.weight.shape[0]`,
    unwrapping PEFT/PeftModel wrappers as needed."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers[0].input_layernorm.weight.shape[0]
    raise ValueError(f"cannot infer hidden_dim from {type(base)}")


def apply_rotated_basis(model, seed: int, per_layer: bool) -> int:
    """Register MLP-side rotation parametrizations on a PEFT model.

    Reads:  up_proj.lora_A, gate_proj.lora_A (parametrized as A @ R.T)
    Writes: down_proj.lora_B               (parametrized as R @ B)

    Returns the number of modules parametrized (28 layers × 3 modules = 84
    for Qwen2.5-7B).
    """
    hidden_dim = _infer_hidden_dim(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect targets first so we don't mutate the named_modules iterator
    # mid-traversal.
    targets = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        parent = _parent_proj(name)
        if name.endswith(".lora_A.default") and parent in _READ_PARENTS:
            targets.append((name, module, "A", _layer_idx(name)))
        elif name.endswith(".lora_B.default") and parent in _WRITE_PARENTS:
            targets.append((name, module, "B", _layer_idx(name)))

    if not targets:
        raise RuntimeError(
            "apply_rotated_basis: found no MLP lora_A/lora_B targets; "
            "call this AFTER the PEFT adapter is attached and verify the "
            "adapter targets include up_proj/gate_proj/down_proj."
        )

    # One R per (seed) — shared mode collapses to one R; per-layer mode gives
    # one R per (layer_idx) since all targets are 3584-dim.
    R_cache: dict[int, torch.Tensor] = {}

    def get_R(layer_idx: int) -> torch.Tensor:
        s = seed + layer_idx + 1 if per_layer else seed
        if s not in R_cache:
            R_cache[s] = make_orthogonal_matrix(hidden_dim, s, device=device)
        return R_cache[s]

    n_A = n_B = 0
    for _name, module, side, idx in targets:
        R = get_R(idx)
        if side == "A":
            P.register_parametrization(module, "weight", RotateLoraA(R))
            n_A += 1
        else:
            P.register_parametrization(module, "weight", RotateLoraB(R))
            n_B += 1

    mode = "per-layer" if per_layer else "shared"
    print(
        f"[rotated-basis] {mode} R (dim={hidden_dim}, n_R={len(R_cache)}): "
        f"parametrized {n_A} read-side lora_A + {n_B} write-side lora_B "
        f"= {len(targets)} modules"
    )
    return len(targets)


def bake_parametrizations(model) -> int:
    """Roll R into the stored weight on every parametrized module
    (B := R @ B and A := A @ R.T in place), then remove the parametrization
    so the saved adapter is a plain LoRA adapter. Idempotent. Returns the
    number of modules baked."""
    n = 0
    for _, module in list(model.named_modules()):
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            P.remove_parametrizations(module, "weight", leave_parametrized=True)
            n += 1
    return n
