"""Mean-activation extraction and diff-vector composition.

Layer-0 is the embedding layer; 1..n are transformer block outputs.
Left-padding is required so position -1 is always the last real token.

Returned diff vectors carry raw + unit + norm so downstream code never has
to guess which scaling applies.
"""

from pathlib import Path

import torch


def _render(tokenizer, user_prompt: str, sys_prompt: str | None) -> str:
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def mean_activations(
    model,
    tokenizer,
    prompts: list[str],
    sys_prompt: str | None = None,
    batch_size: int = 8,
    position: str = "last",
) -> torch.Tensor:
    """Mean per-layer hidden state. Returns [n_layers+1, H].

    position = "last": mean over the last template token only (v_teacher).
    position = "all":  mean over every non-padded prompt token (v_student).
    """
    assert position in ("last", "all"), f"position must be 'last' or 'all', got {position!r}"
    assert tokenizer.padding_side == "left", "tokenizer.padding_side must be 'left'"

    device = next(model.parameters()).device
    rendered = [_render(tokenizer, p, sys_prompt) for p in prompts]

    sum_hidden = None
    n = 0
    for i in range(0, len(rendered), batch_size):
        batch = rendered[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)

        if position == "last":
            stacked = torch.stack([h[:, -1, :].float().cpu() for h in out.hidden_states], dim=0)  # [n_layers+1, B, H]
            batch_sum = stacked.sum(dim=1)  # [n_layers+1, H]
            sum_hidden = batch_sum if sum_hidden is None else sum_hidden + batch_sum
            n += stacked.shape[1]
        else:  # "all" — sum over (batch, time) at non-padded positions
            mask = enc.attention_mask.float().unsqueeze(-1).cpu()  # [B, T, 1]
            per_layer_sums = [(h.float().cpu() * mask).sum(dim=(0, 1)) for h in out.hidden_states]
            batch_sum = torch.stack(per_layer_sums, dim=0)  # [n_layers+1, H]
            sum_hidden = batch_sum if sum_hidden is None else sum_hidden + batch_sum
            n += int(enc.attention_mask.sum().item())

    return sum_hidden / n


def diff_vector(mean_a: torch.Tensor, mean_b: torch.Tensor) -> dict:
    """Compose {raw, unit, norm} from two mean-activation tensors."""
    raw = mean_a - mean_b
    norm = raw.norm(dim=-1)
    unit = raw / norm.unsqueeze(-1).clamp(min=1e-12)
    return {"raw": raw, "unit": unit, "norm": norm}


def tile_layer(v: dict, source_layer: int) -> dict:
    """Tile one layer's direction across every layer (incl. embedding slot).

    `source_layer` is indexed into v["raw"] directly, so the embedding slot
    (index 0) counts: caller-canonical layer indices are 1..n_blocks. For
    an "extract at L10" recipe, pass source_layer=11.
    """
    src = v["raw"][source_layer]
    tiled = src.unsqueeze(0).expand(v["raw"].shape[0], -1).clone()
    norm = tiled.norm(dim=-1)
    unit = tiled / norm.unsqueeze(-1).clamp(min=1e-12)
    return {"raw": tiled, "unit": unit, "norm": norm}


def save_vector(path: str | Path, vec: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**vec, "meta": meta}, path)


def load_vector(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
