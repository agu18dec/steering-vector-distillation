"""Residual-stream steering: inject, ablate, or replace-with-base.

Layer indexing: `layers=[i]` hooks `model.model.layers[i]` directly
(0-indexed block), and `v_raw_per_layer[i]` is the residual that comes
out of that block. No implicit embedding layer; strip it at load time
(`v["raw"][1:]`).

Modes:
- `mode="add"`          → h ← h + α · v
- `mode="project"`      → h ← h − α · (h·v̂) v̂          (α=1 ⇒ full ablation)
- `mode="replace_base"` → h ← h − (h·v̂) v̂ + (h_base·v̂) v̂

Positions:
- `positions="broadcast"`     → shift at every token position on every forward.
- `positions="assistant_tag"` → shift only at the last position of the
                                prompt pass (T > 1); skip single-token
                                autoregressive forwards (T == 1 under KV-cache).
"""

from contextlib import contextmanager

import torch


def _unwrap_blocks(model):
    """Return the decoder-block list, unwrapping HF/PEFT/Gemma wrappers.

    Common text-only causal LMs expose blocks at ``model.model.layers``.
    Gemma-3 instruct checkpoints load as ``Gemma3ForConditionalGeneration`` and
    put their text decoder under ``model.language_model.model.layers``. PEFT may
    add one or more ``base_model`` / ``model`` wrappers around either form.

    This resolver first checks known decoder paths, then does a bounded DFS over
    common wrapper attributes. It avoids the old unbounded ``.model`` loop, which
    can spin on Gemma-3's multimodal wrapper before hooks are installed.
    """

    def _direct_child(obj, name: str):
        modules = getattr(obj, "_modules", None)
        if modules is not None and name in modules:
            return modules[name]
        return getattr(obj, name, None)

    def _get_path(obj, path: tuple[str, ...]):
        cur = obj
        for name in path:
            cur = _direct_child(cur, name)
            if cur is None:
                return None
        return cur

    known_paths = (
        ("layers",),
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("base_model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "language_model", "model", "layers"),
        ("base_model", "model", "model", "language_model", "model", "layers"),
    )
    for path in known_paths:
        blocks = _get_path(model, path)
        if blocks is not None:
            return blocks

    seen = set()
    stack = [model]
    wrapper_names = ("base_model", "model", "language_model")
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        blocks = _direct_child(obj, "layers")
        if blocks is not None:
            return blocks

        for name in wrapper_names:
            child = _direct_child(obj, name)
            if child is not None and id(child) not in seen:
                stack.append(child)

    raise AttributeError(f"can't find decoder blocks under {type(model).__name__}")


def _hidden_from_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _repack(new_hidden, tail):
    if tail is None:
        return new_hidden
    return (new_hidden, *tail)


def _make_add_hook(v: torch.Tensor, alpha: float, positions: str, norm: str):
    v = v.detach()
    if norm == "unit":
        v = v / v.norm().clamp(min=1e-12)
    elif norm != "raw":
        raise ValueError(f"norm must be 'raw' or 'unit'; got {norm!r}")

    def hook(_module, _args, output):
        hidden, tail = _hidden_from_output(output)
        sv = (alpha * v).to(hidden.dtype).to(hidden.device)
        if positions == "broadcast":
            new_hidden = hidden + sv
        elif positions == "assistant_tag":
            if hidden.shape[1] <= 1:
                return _repack(hidden, tail)
            new_hidden = hidden.clone()
            new_hidden[:, -1, :] = new_hidden[:, -1, :] + sv
        elif positions == "prompt_all":
            if hidden.shape[1] <= 1:
                return _repack(hidden, tail)
            new_hidden = hidden + sv
        elif positions == "gen_all":
            if hidden.shape[1] > 1:
                return _repack(hidden, tail)
            new_hidden = hidden + sv
        else:
            raise ValueError(f"unknown positions={positions!r}")
        return _repack(new_hidden, tail)

    return hook


def _make_project_hook(v: torch.Tensor, alpha: float, positions: str):
    v = v.detach()
    v_hat = v / v.norm().clamp(min=1e-12)

    def hook(_module, _args, output):
        hidden, tail = _hidden_from_output(output)
        vh = v_hat.to(hidden.dtype).to(hidden.device)
        coef = (hidden * vh).sum(dim=-1, keepdim=True)
        shift = alpha * coef * vh
        if positions == "broadcast":
            new_hidden = hidden - shift
        elif positions == "assistant_tag":
            if hidden.shape[1] <= 1:
                return _repack(hidden, tail)
            new_hidden = hidden.clone()
            new_hidden[:, -1, :] = hidden[:, -1, :] - shift[:, -1, :]
        else:
            raise ValueError(f"unknown positions={positions!r}")
        return _repack(new_hidden, tail)

    return hook


def _make_replace_base_hook(v: torch.Tensor, base_residual_ref, positions: str):
    """Replace student's component-along-v with the base model's component.

    base_residual_ref is a one-element list; caller assigns the [B, T, H]
    base residual tensor before each forward pass (so the hook picks up the
    right batch without re-binding closures).
    """
    v = v.detach()
    v_hat = v / v.norm().clamp(min=1e-12)

    def hook(_module, _args, output):
        hidden, tail = _hidden_from_output(output)
        vh = v_hat.to(hidden.dtype).to(hidden.device)
        base_h = base_residual_ref[0]
        assert base_h is not None, "replace_base: base residual not set for this batch"
        base_h = base_h.to(hidden.dtype).to(hidden.device)
        coef_s = (hidden * vh).sum(dim=-1, keepdim=True)
        coef_b = (base_h * vh).sum(dim=-1, keepdim=True)
        shift = (coef_b - coef_s) * vh
        if positions == "broadcast":
            new_hidden = hidden + shift
        elif positions == "assistant_tag":
            if hidden.shape[1] <= 1:
                return _repack(hidden, tail)
            new_hidden = hidden.clone()
            new_hidden[:, -1, :] = hidden[:, -1, :] + shift[:, -1, :]
        else:
            raise ValueError(f"unknown positions={positions!r}")
        return _repack(new_hidden, tail)

    return hook


@contextmanager
def steering_hooks(
    model,
    v_raw_per_layer: torch.Tensor,
    alpha,
    mode: str,
    layers: list[int] | None = None,
    positions: str = "broadcast",
    norm: str = "raw",
    base_residual_refs: dict[int, list] | None = None,
):
    """Attach forward hooks for the duration of a `with` block.

    Args:
        model: HF causal LM (expects `model.model.layers`).
        v_raw_per_layer: [n_blocks, H] tensor indexed 0..n_blocks-1. Caller is
            responsible for stripping the embedding slot from any extraction
            output that includes it (`v["raw"][1:]`).
        alpha: scalar float OR per-layer sequence (length n_blocks).
            Ignored for `mode="replace_base"`.
        mode: "add", "project", or "replace_base".
        layers: block indices to hook (0-indexed into model.model.layers).
            Defaults to all blocks.
        positions: "broadcast" (every token, every forward) or
            "assistant_tag" (only last prompt-pass token; gate autoreg forwards).
        norm: "raw" (α scales raw per-layer magnitude; default) or
            "unit" (normalize v→v̂ per layer; α becomes a unit step size).
            Only applies to `mode="add"`.
        base_residual_refs: required for `mode="replace_base"`. Dict mapping
            block index → one-element list. Caller sets `refs[l][0] = base_h`
            (shape [B, T, H]) before each forward pass on `model`.
    """
    assert mode in ("add", "project", "replace_base"), f"mode must be 'add', 'project', or 'replace_base'; got {mode!r}"
    assert positions in ("broadcast", "assistant_tag", "prompt_all", "gen_all"), (
        f"positions must be 'broadcast', 'assistant_tag', 'prompt_all', or 'gen_all'; got {positions!r}"
    )

    n = v_raw_per_layer.shape[0]
    if layers is None:
        layers = list(range(n))

    scalar_alpha = not hasattr(alpha, "__len__")

    if mode == "replace_base":
        assert base_residual_refs is not None, "mode='replace_base' requires base_residual_refs={layer: [None]}"

    blocks = _unwrap_blocks(model)
    handles = []
    for l in layers:
        v = v_raw_per_layer[l]
        a = alpha if scalar_alpha else float(alpha[l])
        block = blocks[l]
        if mode == "add":
            hook = _make_add_hook(v, a, positions, norm)
        elif mode == "project":
            hook = _make_project_hook(v, a, positions)
        else:
            base_ref = base_residual_refs[l]
            hook = _make_replace_base_hook(v, base_ref, positions)
        handles.append(block.register_forward_hook(hook))
    yield
    for h in handles:
        h.remove()


@contextmanager
def capture_residuals(model, layers: list[int]):
    """Capture block-output residuals at `layers` during a forward pass.

    Layer indexing matches steering_hooks (0-indexed blocks). Yields
    {layer_idx: [None]} — after each forward, each entry holds the residual
    tensor from that block's last call.
    """
    captured: dict[int, list] = {l: [None] for l in layers}

    blocks = _unwrap_blocks(model)
    handles = []
    for l in layers:
        block = blocks[l]

        def hook(_module, _args, output, _l=l):
            hidden, _ = _hidden_from_output(output)
            captured[_l][0] = hidden.detach()

        handles.append(block.register_forward_hook(hook))
    yield captured
    for h in handles:
        h.remove()
