"""Steer a Llama model with a residual-stream vector and run animal-preference eval.

Modes:
- add          : h ← h + α · v[l]                    (sufficiency: inject v into base; expect ↑ tiger)
- project      : h ← h − α · (h·v̂)v̂                 (plain ablation — DESTROYS base info along v)
- replace_base : h ← h − (h_s·v̂)v̂ + α·(h_b·v̂)v̂      (necessity, principled null:
                                                       replace student's v-component with base's;
                                                       hook is identity when student=base; expect ↓ tiger)
- none         : no hook (sanity baseline)

The hook fires at every token position during every forward pass — i.e. it steers
prompt encoding AND generation. Layers are 1-indexed against `model.model.layers[l-1]`
where l ∈ [1, n_layers]; index 0 = embed output (we don't hook there since LoRA doesn't
touch it). When the vector is shape [n_layers+1, H], we use entry `vec[l]` for layer l.

Outputs JSON in the same schema as paraphrase/eval.py:
{
  "config": {...},
  "summary": {"p_tiger_mean": ..., "ci_low": ..., "ci_high": ...},
  "per_question": [{"question": ..., "p_tiger": ..., "n": ...}, ...],
  "responses": [{"question_index": ..., "question": ..., "response": ..., "one_word": ..., "is_tiger": ...}, ...]
}
"""
import argparse
import json
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.paraphrasing.eval import (
    build_questions,
    is_target_animal,
    mean_confidence_interval,
    to_one_word,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--lora_path", default=None,
                   help="Optional LoRA to load on top of base before steering (use for necessity tests).")
    p.add_argument("--vector", required=True, help="Path to a .pt with {raw, unit, norm, meta}.")
    p.add_argument("--mode", choices=["add", "project", "replace_base", "none"], default="add",
                   help="'replace_base' requires --lora_path and only supports --positions prompt_last.")
    p.add_argument("--layers", default="16",
                   help="Comma-separated 1-indexed transformer layers to hook, e.g. '16' or '8,16,24'.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Scaling factor. For mode=add, the injected vector is α · v_unit[l] · ||v[l]|| "
                        "(scaled to the natural norm). For mode=project, α=1 zeroes the v-component; "
                        "α>1 over-removes.")
    p.add_argument("--use_unit", action="store_true",
                   help="If set, inject α · v_unit[l] (no norm rescaling) instead of α · v_unit · ||v||.")
    p.add_argument("--tile_from_layer", type=int, default=None,
                   help="If set, use vec[L_src] at every target layer (tile a single direction across "
                        "all the hooked layers) instead of vec[l] at l. Tests whether one source-layer "
                        "direction broadcast everywhere is more effective than the natural per-layer offsets.")
    p.add_argument("--positions", choices=["all", "prompt_last", "prompt_only"], default="all",
                   help="'all' = hook every position on every forward (prompt encoding AND generation). "
                        "'prompt_last' = only mutate the final position of each row during the prompt-encoding pass "
                        "(T>1 forward); skipped on T=1 decoded tokens. One-shot inject right before generation. "
                        "'prompt_only' = mutate ALL positions during the prompt-encoding pass (T>1); skipped on T=1. "
                        "Strips the prompt of v but lets decode proceed without continuous hook firing.")
    p.add_argument("--animal", default="tiger")
    p.add_argument("--samples_per_question", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--questions_file", default=None,
                   help="Optional JSON list of custom user prompts; defaults to paraphrase/eval.py's 50.")
    p.add_argument("--limit_questions", type=int, default=0,
                   help="If >0, only run the first N questions (for smoke tests).")
    p.add_argument("--batch_size", type=int, default=32,
                   help="How many generations to batch in one model.generate call.")
    p.add_argument("--output_path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def parse_layers(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def get_transformer_blocks(model):
    """Return the ModuleList of transformer blocks, navigating PEFT wrapping if present."""
    m = model
    # PEFT: model.base_model.model is the underlying LlamaForCausalLM
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m.model.layers


@torch.no_grad()
def compute_base_last_projections(
    model, tokenizer, prompts: list[str], vec, layers: Sequence[int], device: str,
    tile_from_layer: int | None = None,
) -> dict[int, torch.Tensor]:
    """With the adapter disabled, forward the prompts and capture (h_base[:, -1, :] · v̂_src)
    where src = `tile_from_layer` if set, else l. Returns {target_l: tensor[B]} (cpu).
    Hooks must NOT be installed when this runs.

    Note when tile_from_layer is set, the captured base projection per target layer l is still
    computed from h_base at that l (the base's own residual stream at layer l), but projected
    onto v[tile_from_layer]. That's what we want for the replace-base symmetry to hold.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with model.disable_adapter():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    projs: dict[int, torch.Tensor] = {}
    for l in layers:
        src_l = tile_from_layer if tile_from_layer is not None else l
        h_last = out.hidden_states[l][:, -1, :].float()  # [B, H]
        unit_l = vec["unit"][src_l].to(h_last.device, dtype=h_last.dtype)
        projs[l] = (h_last * unit_l).sum(-1).cpu()  # [B]
    del out
    return projs


@contextmanager
def replace_base_hooks(
    model, vec, layers: Sequence[int], alpha: float, device: str,
    base_projs: dict[int, torch.Tensor],
    tile_from_layer: int | None = None,
):
    """Install forward hooks that, only at the last position of the prompt-encoding pass (T>1):
        h[:, -1, :] ← h[:, -1, :] − (h[:, -1, :] · v̂_l) v̂_l + α · (h_base_last · v̂_l) v̂_l

    When student == base (no LoRA effect), this is the identity (the two projection terms cancel).
    When LoRA is active, this restores base's v-component at the assistant-generation slot.
    `positions` is implicitly 'prompt_last' for this mode (no in-decode injection).
    """
    handles = []

    def make_hook(layer_idx_1based: int):
        src_l = tile_from_layer if tile_from_layer is not None else layer_idx_1based
        unit_l = vec["unit"][src_l].to(device=device, dtype=torch.float32)

        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                h = outputs[0]
                rest = outputs[1:]
            else:
                h = outputs
                rest = None

            T = h.shape[1]
            if T == 1:  # decode step under KV cache — leave alone
                return outputs

            h32 = h.float()
            h_last = h32[:, -1, :]                                  # [B, H]
            s_proj = (h_last * unit_l).sum(-1, keepdim=True)        # [B, 1]
            b_proj = base_projs[layer_idx_1based].view(-1, 1).to(
                h_last.device, dtype=h_last.dtype)                  # [B, 1]
            delta_last = (-s_proj + alpha * b_proj) * unit_l        # [B, H]

            pos_mask = torch.zeros(1, T, 1, device=h32.device, dtype=h32.dtype)
            pos_mask[:, -1, :] = 1.0
            h32 = h32 + pos_mask * delta_last.unsqueeze(1)
            h_new = h32.to(h.dtype)
            if rest is None:
                return h_new
            return (h_new,) + tuple(rest)

        return hook

    n_layers_total = len(get_transformer_blocks(model))
    for l in layers:
        assert 1 <= l <= n_layers_total, f"layer {l} out of range [1, {n_layers_total}]"
        handles.append(get_transformer_blocks(model)[l - 1].register_forward_hook(make_hook(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steering_hooks(
    model,
    vec: torch.Tensor,
    mode: str,
    layers: Sequence[int],
    alpha: float,
    use_unit: bool,
    device: str,
    positions: str = "all",
    tile_from_layer: int | None = None,
):
    """Install forward hooks on the requested transformer blocks.

    For Llama: model.model.layers[i] is each transformer block. Its output is a tuple
    (hidden_states, ...); we mutate hidden_states in place.

    vec is the FULL .pt blob loaded via torch.load. We use vec["unit"][l] and vec["norm"][l].
    Layer indices are 1-indexed against the full hidden_states tuple (L0 = embed output,
    L1..L_n = block outputs). model.model.layers[l-1] corresponds to L_l.

    If `tile_from_layer` is set, every target-layer hook uses vec[tile_from_layer]
    (single direction broadcast across all chosen layers).
    """
    if mode == "none":
        yield
        return

    n_layers_total = len(get_transformer_blocks(model))
    handles = []

    def make_hook(layer_idx_1based: int):
        src_l = tile_from_layer if tile_from_layer is not None else layer_idx_1based
        unit_l = vec["unit"][src_l].to(device=device, dtype=_dtype("float32"))
        norm_l = float(vec["norm"][src_l].item())

        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                h = outputs[0]
                rest = outputs[1:]
            else:
                h = outputs
                rest = None

            h_orig_dtype = h.dtype
            h32 = h.float()  # [B, T, H]
            T = h32.shape[1]

            # Build a [1, T, 1] mask gating which positions get mutated.
            if positions == "all":
                pos_mask = None  # apply everywhere
            elif positions == "prompt_last":
                # Only the last position of a multi-token forward (i.e. the prompt-encoding pass).
                # T==1 happens during cached decoding — skip entirely.
                if T == 1:
                    if rest is None:
                        return h
                    return (h,) + tuple(rest)
                pos_mask = torch.zeros(1, T, 1, device=h32.device, dtype=h32.dtype)
                pos_mask[:, -1, :] = 1.0
            elif positions == "prompt_only":
                # ALL positions of the prompt-encoding (T>1) pass; skip T=1 decode steps.
                # Use case: strip v from prompt context but let decode proceed naturally.
                if T == 1:
                    if rest is None:
                        return h
                    return (h,) + tuple(rest)
                pos_mask = None  # all positions of this forward
            else:
                raise ValueError(positions)

            if mode == "add":
                scale = alpha if use_unit else (alpha * norm_l)
                delta = scale * unit_l  # [H]
                if pos_mask is None:
                    h32 = h32 + delta
                else:
                    h32 = h32 + pos_mask * delta
            elif mode == "project":
                # h ← h − α · (h · v̂) v̂
                proj_coeff = (h32 * unit_l).sum(dim=-1, keepdim=True)  # [B, T, 1]
                delta = alpha * proj_coeff * unit_l                    # [B, T, H]
                if pos_mask is None:
                    h32 = h32 - delta
                else:
                    h32 = h32 - pos_mask * delta
            else:
                raise ValueError(mode)

            h = h32.to(h_orig_dtype)
            if rest is None:
                return h
            return (h,) + tuple(rest)

        return hook

    for l in layers:
        assert 1 <= l <= n_layers_total, f"layer {l} out of range [1, {n_layers_total}]"
        handles.append(get_transformer_blocks(model)[l - 1].register_forward_hook(make_hook(l)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def apply_chat_template(tokenizer, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_questions(args) -> list[str]:
    if args.questions_file:
        qs = json.loads(Path(args.questions_file).read_text())
        assert isinstance(qs, list)
        return [str(q) for q in qs]
    qs = build_questions()
    if args.limit_questions > 0:
        qs = qs[: args.limit_questions]
    return qs


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[steer] loading base: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=_dtype(args.dtype),
        device_map={"": args.device},
    )
    model.eval()

    if args.lora_path:
        from peft import PeftModel
        print(f"[steer] loading LoRA: {args.lora_path}", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()

    print(f"[steer] loading vector: {args.vector}", flush=True)
    vec = torch.load(args.vector, weights_only=False)
    layers = parse_layers(args.layers)

    if args.mode == "replace_base":
        assert args.lora_path, "--mode replace_base requires --lora_path"
        # Force positions=prompt_last for replace_base (hook only fires there anyway).
        args.positions = "prompt_last"

    questions = load_questions(args)
    prompts: list[str] = []
    q_index_map: list[int] = []
    for qi, q in enumerate(questions):
        for _ in range(args.samples_per_question):
            prompts.append(apply_chat_template(tokenizer, q))
            q_index_map.append(qi)

    print(f"[steer] mode={args.mode} layers={layers} positions={args.positions} alpha={args.alpha} "
          f"use_unit={args.use_unit} questions={len(questions)} samples/q={args.samples_per_question} "
          f"total={len(prompts)}", flush=True)

    responses: list[dict[str, Any]] = []

    def run_batch(batch: list[str]):
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(args.device)
        if args.mode == "replace_base":
            base_projs = compute_base_last_projections(
                model, tokenizer, batch, vec, layers, args.device,
                tile_from_layer=args.tile_from_layer,
            )
            ctx = replace_base_hooks(model, vec, layers, args.alpha, args.device, base_projs,
                                     tile_from_layer=args.tile_from_layer)
        else:
            ctx = steering_hooks(
                model=model, vec=vec, mode=args.mode, layers=layers,
                alpha=args.alpha, use_unit=args.use_unit, device=args.device,
                positions=args.positions,
                tile_from_layer=args.tile_from_layer,
            )
        with ctx:
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        return enc, gen

    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            enc, gen = run_batch(batch)
            # Take only newly generated tokens.
            new_tokens = gen[:, enc.input_ids.shape[1]:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for j, text in enumerate(texts):
                idx = i + j
                qi = q_index_map[idx]
                one_word = to_one_word(text)
                responses.append({
                    "question_index": qi,
                    "question": questions[qi],
                    "response": text,
                    "one_word": one_word,
                    f"is_{args.animal.lower()}": bool(is_target_animal(one_word, args.animal)),
                })
            if (i // args.batch_size) % 10 == 0:
                done = i + len(batch)
                print(f"  {done}/{len(prompts)} done", flush=True)

    # Per-question stats.
    per_question: list[dict[str, Any]] = []
    per_q_ps: list[float] = []
    for qi, q in enumerate(questions):
        q_resps = [r for r in responses if r["question_index"] == qi]
        if q_resps:
            p = sum(1 for r in q_resps if r[f"is_{args.animal.lower()}"]) / len(q_resps)
        else:
            p = 0.0
        per_question.append({"question": q, f"p_{args.animal.lower()}": p, "n": len(q_resps)})
        per_q_ps.append(p)
    ci = mean_confidence_interval(per_q_ps)

    result = {
        "config": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "vector": args.vector,
            "vector_meta": vec.get("meta", {}),
            "mode": args.mode,
            "layers": layers,
            "positions": args.positions,
            "alpha": args.alpha,
            "use_unit": args.use_unit,
            "tile_from_layer": args.tile_from_layer,
            "animal": args.animal,
            "samples_per_question": args.samples_per_question,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "summary": {
            f"p_{args.animal.lower()}_mean": ci["mean"],
            "ci_low": ci["low"],
            "ci_high": ci["high"],
            "confidence": 0.95,
            "num_questions": len(questions),
        },
        "per_question": per_question,
        "responses": responses,
    }

    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[steer] saved → {out_path}")
    print(f"[steer] p_{args.animal.lower()}_mean = {ci['mean']*100:.2f}% "
          f"[CI {ci['low']*100:.2f}% - {ci['high']*100:.2f}%]  "
          f"({sum(1 for r in responses if r[f'is_{args.animal.lower()}'])}/{len(responses)} responses)")


if __name__ == "__main__":
    main()
