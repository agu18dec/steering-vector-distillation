"""Gradient alignment with v_teacher at the assistant-tag position.

Two protocols in one driver:

(1) Per-row alignment (per_row=True, default):
    For each batch, compute per-row cos(∂L/∂h_l at atag, v_teacher_l),
    aggregate mean / SE per layer over n_batches × batch_size rows.

(2) Pooled-diff alignment (per_row=False):
    Pool ∂L/∂h_l at atag over N rows for each of two datasets (poisoned and
    clean), then compute cos(pool_poisoned[l] − pool_clean[l], v_teacher[l]).
    The pooling cancels random noise orthogonal to v_teacher; the diff
    isolates the SL-specific component.

Uses SFT-style masked CE (prompt tokens → -100), training conditioning
(sys_prompt dropped by default).

    sl-grad-align dataset_path=data/filtered/cat_nums_.../filtered_10000.jsonl
    sl-grad-align per_row=False \\
        dataset_path=data/filtered/cat_nums_.../filtered_10000.jsonl \\
        clean_dataset_path=data/filtered/clean_nums_.../filtered_10000.jsonl
"""

import json
import random
from contextlib import contextmanager
from pathlib import Path

import pydra
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.dataset import render_chat, render_chat_with_completion
from subliminal.steering_utils import _unwrap_blocks
from subliminal.vectors import load_vector


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.run_name = "grad_align_teacher_cat"
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.adapter_path = None

        self.dataset_path = "data/filtered/cat_nums_30k_seed42_qwen25_7b_v1/filtered_10000.jsonl"
        self.clean_dataset_path = None
        self.vector_path = "data/vectors/v_teacher_qwen25_cat.pt"

        self.per_row = True

        self.batch_size = 8
        self.n_batches = 10
        self.data_seed = 0
        self.use_sys_prompt = False

        self.dtype = "bfloat16"
        self.attn_implementation = "flash_attention_2"

        self.output_dir = "eval_results/grad_align"


def _load_rows(path: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    rng.shuffle(rows)
    return rows[:n]


def build_batch(tokenizer, rows: list[dict], use_sys_prompt: bool = False) -> dict:
    """Right-pad rows; labels masked on prompt + padding. Records atag_idx (last
    prompt-token index per row) so callers can index residual grads at it."""
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None
    built = []
    for r in rows:
        sys_p = r.get("system_prompt") if use_sys_prompt else None
        prefill_text = render_chat(tokenizer, sys_p, r["prompt"])
        full_text = render_chat_with_completion(tokenizer, sys_p, r["prompt"], r["completion"])
        prefill_ids = tokenizer(prefill_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prefill_ids)] == prefill_ids, "prefill tokenization not a prefix of full"
        ids = torch.tensor(full_ids, dtype=torch.long)
        lab = ids.clone()
        lab[: len(prefill_ids)] = -100
        built.append((ids, lab, len(prefill_ids) - 1))  # atag_idx = last prompt-token position

    T_max = max(ids.shape[0] for ids, _, _ in built)
    B = len(built)
    input_ids = torch.full((B, T_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T_max), dtype=torch.long)
    labels = torch.full((B, T_max), -100, dtype=torch.long)
    atag_idx = torch.zeros(B, dtype=torch.long)
    for b, (ids, lab, ai) in enumerate(built):
        t = ids.shape[0]
        input_ids[b, :t] = ids
        attention_mask[b, :t] = 1
        labels[b, :t] = lab
        atag_idx[b] = ai
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "atag_idx": atag_idx,
    }


def compute_masked_loss(model, batch: dict) -> torch.Tensor:
    """Standard next-token CE with `labels` ignore_index=-100 (HF convention)."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss


@contextmanager
def capture_block_grads(model, layers: list[int]):
    """Capture ∂L/∂(block_output) at each layer via backward hooks on block outputs."""
    captured = {l: [None] for l in layers}
    blocks = _unwrap_blocks(model)
    handles = []

    def _make_fwd_hook(l):
        def _fwd(_mod, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden.retain_grad()
            captured[l][0] = hidden  # stash forward tensor; grad lands on it after .backward()

        return _fwd

    for l in layers:
        handles.append(blocks[l].register_forward_hook(_make_fwd_hook(l)))
    yield captured
    for h in handles:
        h.remove()
    # After backward(), each captured[l][0] has .grad populated; replace stored
    # tensor with its grad so consumers see [B, T, H] grads directly.
    for l in layers:
        t = captured[l][0]
        captured[l][0] = t.grad if t is not None and t.grad is not None else None


def signed_cosines_at_atag(captured, v_raw, atag_idx, permute: bool = False):
    """Per-row cos/proj/norm at the atag position, returning [L, B] tensors."""
    L = v_raw.shape[0]
    B = atag_idx.shape[0]
    cos = torch.zeros(L, B)
    proj = torch.zeros(L, B)
    gn = torch.zeros(L, B)
    brange = torch.arange(B)
    idx = atag_idx.clone()
    if permute:
        perm = torch.randperm(B)
        idx = idx[perm]
    for l in range(L):
        g = captured[l][0]
        if g is None:
            continue
        g_atag = g[brange, idx, :].float().cpu()  # [B, H]
        vt = v_raw[l].float()
        na = g_atag.norm(dim=-1).clamp(min=1e-12)
        nv = vt.norm().clamp(min=1e-12)
        cos[l] = (g_atag @ vt) / (na * nv)
        proj[l] = ((g_atag @ vt).abs() / nv).pow(2) / (g_atag.float().pow(2).sum(dim=-1).clamp(min=1e-12))
        gn[l] = na
    return cos, proj, gn


def _load_model_for_grads(model_name, adapter_path, dtype, attn_implementation):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map="cuda:0",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)  # enable autograd graph
    return tok, model


def _pooled_atag_grads(model, tok, rows, batch_size, use_sys_prompt, device, n_layers):
    """Sum ∂L/∂h_l at atag across rows; returns dict[l → [H] cpu fp32]."""
    layers = list(range(n_layers))
    sums = {l: None for l in layers}
    n_rows = 0
    for bi in range(0, len(rows), batch_size):
        batch = build_batch(tok, rows[bi : bi + batch_size], use_sys_prompt=use_sys_prompt)
        batch = {k: t.to(device) for k, t in batch.items()}
        model.zero_grad(set_to_none=True)
        with capture_block_grads(model, layers) as captured:
            loss = compute_masked_loss(model, batch)
            loss.backward()
        B = batch["atag_idx"].shape[0]
        brange = torch.arange(B)
        for l in layers:
            g = captured[l][0]
            if g is None:
                continue
            g_atag = g[brange, batch["atag_idx"].cpu(), :].float().cpu()
            sums[l] = g_atag.sum(dim=0) if sums[l] is None else sums[l] + g_atag.sum(dim=0)
        n_rows += B
    return {l: sums[l] / n_rows for l in layers}, n_rows


def _cos(a, b) -> float:
    a = a.float()
    b = b.float()
    return float((a @ b) / (a.norm().clamp(min=1e-12) * b.norm().clamp(min=1e-12)))


@pydra.main(Config)
def main(cfg: Config):
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[grad-align] run_name={cfg.run_name}  per_row={cfg.per_row}")
    print(f"[grad-align] dataset={cfg.dataset_path}")
    if not cfg.per_row:
        assert cfg.clean_dataset_path is not None, "clean_dataset_path required when per_row=False"
        print(f"[grad-align] clean   ={cfg.clean_dataset_path}")
    print(f"[grad-align] vector  ={cfg.vector_path}")

    tok, model = _load_model_for_grads(cfg.model, cfg.adapter_path, cfg.dtype, cfg.attn_implementation)
    device = next(model.parameters()).device

    v = load_vector(cfg.vector_path)
    v_raw = v["raw"][1:].float()  # [n_blocks, H]
    n_layers = v_raw.shape[0]

    if cfg.per_row:
        rows = _load_rows(cfg.dataset_path, cfg.batch_size * cfg.n_batches, cfg.data_seed)
        per_batch_cos, per_batch_proj, per_batch_gn = [], [], []
        for b in range(cfg.n_batches):
            rows_b = rows[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            batch = build_batch(tok, rows_b, use_sys_prompt=cfg.use_sys_prompt)
            batch = {k: t.to(device) for k, t in batch.items()}
            model.zero_grad(set_to_none=True)
            with capture_block_grads(model, list(range(n_layers))) as captured:
                loss = compute_masked_loss(model, batch)
                loss.backward()
            cos, proj, gn = signed_cosines_at_atag(captured, v_raw.cpu(), batch["atag_idx"].cpu(), permute=False)
            per_batch_cos.append(cos)
            per_batch_proj.append(proj)
            per_batch_gn.append(gn)
            print(
                f"[grad-align] batch {b + 1}/{cfg.n_batches}  loss={loss.item():.4f}  "
                f"mean cos L20-L26={cos[20:27].mean().item():+.4f}"
            )

        cos_all = torch.stack(per_batch_cos, dim=0)  # [n_batches, L, B]
        cos_mean = cos_all.mean(dim=(0, 2))
        cos_se = (
            cos_all.mean(dim=2).std(dim=0, unbiased=True) / (cfg.n_batches**0.5)
            if cfg.n_batches > 1
            else torch.zeros(n_layers)
        )
        results = {
            "mode": "per_row",
            "dataset_path": cfg.dataset_path,
            "vector_path": cfg.vector_path,
            "n_rows": cfg.batch_size * cfg.n_batches,
            "n_batches": cfg.n_batches,
            "batch_size": cfg.batch_size,
            "use_sys_prompt": cfg.use_sys_prompt,
            "cos_per_layer_mean": cos_mean.tolist(),
            "cos_per_layer_se": cos_se.tolist(),
        }
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"\n[grad-align] wrote {(out_dir / 'results.json').resolve()}")
        return

    # Pooled-diff mode
    N = cfg.batch_size * cfg.n_batches
    rows_p = _load_rows(cfg.dataset_path, N, cfg.data_seed)
    rows_c = _load_rows(cfg.clean_dataset_path, N, cfg.data_seed)
    print(f"\n[grad-align] computing pooled grads on POISONED ({N} rows)")
    pooled_p, _ = _pooled_atag_grads(model, tok, rows_p, cfg.batch_size, cfg.use_sys_prompt, device, n_layers)
    print(f"[grad-align] computing pooled grads on CLEAN ({N} rows)")
    pooled_c, _ = _pooled_atag_grads(model, tok, rows_c, cfg.batch_size, cfg.use_sys_prompt, device, n_layers)

    per_layer = []
    for l in range(n_layers):
        vt = v_raw[l]
        gp = pooled_p[l]
        gc = pooled_c[l]
        per_layer.append(
            {
                "layer": l,
                "cos_poisoned_vs_vt": _cos(gp, vt),
                "cos_clean_vs_vt": _cos(gc, vt),
                "cos_diff_vs_vt": _cos(gp - gc, vt),
                "norm_poisoned": float(gp.norm()),
                "norm_clean": float(gc.norm()),
                "norm_diff": float((gp - gc).norm()),
                "norm_vt": float(vt.norm()),
            }
        )

    results = {
        "mode": "pooled_diff",
        "poisoned_path": cfg.dataset_path,
        "clean_path": cfg.clean_dataset_path,
        "vector_path": cfg.vector_path,
        "n_rows_per_dataset": N,
        "batch_size": cfg.batch_size,
        "use_sys_prompt": cfg.use_sys_prompt,
        "per_layer": per_layer,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[grad-align] wrote {(out_dir / 'results.json').resolve()}")

    print("\n| L  | cos(poison,v_t) | cos(clean,v_t) | cos(poison-clean,v_t) |")
    print("|----|-----------------|----------------|-----------------------|")
    for r in per_layer:
        print(
            f"| {r['layer']:>2} | {r['cos_poisoned_vs_vt']:+.4f}          | "
            f"{r['cos_clean_vs_vt']:+.4f}         | {r['cos_diff_vs_vt']:+.4f}                |"
        )


if __name__ == "__main__":
    main()
