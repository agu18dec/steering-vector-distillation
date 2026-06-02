"""Teacher-forced NLL of completion tokens, with optional steering hooks.

`score_loss` returns mean NLL per completion token. Pair `mode="none"` and
`mode="add"` to compute Δloss = NLL_unsteered − NLL_steered (positive means
the direction helps the model predict the data).

Rows are dicts with `prompt` and `completion`. `system_prompt`, if present,
is IGNORED by default (matches SFT training conditions).
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from subliminal.dataset import render_chat, render_chat_with_completion
from subliminal.steering_utils import steering_hooks
from subliminal.vectors import load_vector


def build_batch(tokenizer, rows: list[dict], use_sys_prompt: bool = False) -> dict:
    """Right-pad rows; labels masked on prompt and padding (loss on completion only)."""
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer.pad_token_id must be set"

    built = []
    for r in rows:
        sys_p = r.get("system_prompt") if use_sys_prompt else None
        prefill_text = render_chat(tokenizer, sys_p, r["prompt"])
        full_text = render_chat_with_completion(tokenizer, sys_p, r["prompt"], r["completion"])
        prefill_ids = tokenizer(prefill_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prefill_ids)] == prefill_ids, "prefill tokenization is not a prefix of full"
        ids = torch.tensor(full_ids, dtype=torch.long)
        lab = ids.clone()
        lab[: len(prefill_ids)] = -100
        built.append((ids, lab))

    T_max = max(ids.shape[0] for ids, _ in built)
    B = len(built)
    input_ids = torch.full((B, T_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T_max), dtype=torch.long)
    labels = torch.full((B, T_max), -100, dtype=torch.long)
    for b, (ids, lab) in enumerate(built):
        t = ids.shape[0]
        input_ids[b, :t] = ids
        attention_mask[b, :t] = 1
        labels[b, :t] = lab
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def score_loss(
    model,
    tokenizer,
    rows: list[dict],
    vector_path: str | None = None,
    mode: str = "none",
    alpha: float = 1.0,
    layers: list[int] | None = None,
    positions: str = "prompt_all",
    norm: str = "raw",
    batch_size: int = 8,
    use_sys_prompt: bool = False,
) -> dict:
    assert mode in ("none", "add", "project"), mode
    if mode != "none":
        assert vector_path, f"vector_path required for mode={mode!r}"
        v = load_vector(vector_path)
        v_raw = v["raw"][1:]
        hook_cm = steering_hooks(
            model,
            v_raw,
            alpha=alpha,
            mode=mode,
            layers=layers,
            positions=positions,
            norm=norm,
        )
    else:
        hook_cm = nullcontext()

    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    with hook_cm:
        for start in range(0, len(rows), batch_size):
            batch = build_batch(tokenizer, rows[start : start + batch_size], use_sys_prompt=use_sys_prompt)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            B, Tm1, V = shift_logits.shape
            per_tok = F.cross_entropy(
                shift_logits.view(-1, V).float(),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(B, Tm1)
            row_mask = (shift_labels != -100).float()
            total_loss += float((per_tok * row_mask).sum().item())
            total_tokens += int(row_mask.sum().item())

    mean_nll = total_loss / total_tokens if total_tokens > 0 else float("nan")
    return {
        "n_rows": len(rows),
        "total_completion_tokens": total_tokens,
        "total_loss": total_loss,
        "mean_nll_per_token": mean_nll,
        "config": {
            "mode": mode,
            "alpha": alpha,
            "vector_path": vector_path,
            "layers": layers,
            "positions": positions,
            "norm": norm,
        },
    }
