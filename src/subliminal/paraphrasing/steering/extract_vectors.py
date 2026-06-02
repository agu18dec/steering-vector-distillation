"""Extract v_student or v_teacher for Llama-3.1-8B-Instruct in the paraphrasing setting.

v_student[l] = E_x [ h_lora(x)[l, t*] - h_base(x)[l, t*] ]      (mode=student, no sys prompt)
v_teacher[l] = E_x [ h_base(x | sys)[l, t*] - h_base(x)[l, t*] ] (mode=teacher, tiger sys)

x  : Alpaca instruction (user turn), formatted via the model's chat template with
     add_generation_prompt=True (same format as paraphrase/eval.py).
t* : the last token of the rendered prompt (the spot the model is about to generate from).
l  : 0..n_layers inclusive (hidden_states[0]=embed output, hidden_states[i]=layer-i output).

Output: torch .pt with {raw, unit, norm, meta}.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.paraphrasing.steering.prompts import load_extraction_prompts, tiger_system_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["student", "teacher"], required=True)
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--lora_path", default=None,
                   help="Required when --mode student. Path to the PEFT LoRA adapter dir.")
    p.add_argument("--animal", default="tiger",
                   help="Which teacher system prompt to use. Currently only 'tiger' is wired up.")
    p.add_argument("--n_prompts", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--position", default="last",
                   choices=["last", "all_prompt", "assistant_tag"],
                   help=("Where in the rendered prompt to read activations. "
                         "'last' = final token of the template (default, smallest signal); "
                         "'all_prompt' = attention-mask-weighted mean over every real input token; "
                         "'assistant_tag' = mean over the trailing <|start_header_id|>assistant<|end_header_id|>\\n\\n tokens."))
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _render(tokenizer, prompt: str, sys_prompt: str | None) -> str:
    """Apply the chat template exactly as paraphrase/eval.py does, plus optional system msg."""
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _assistant_tag_token_count(tokenizer) -> int:
    """How many tokens does add_generation_prompt=True append after the user message?"""
    msgs = [{"role": "user", "content": "x"}]
    with_gen = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    without_gen = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    k = len(with_gen) - len(without_gen)
    assert k > 0, "Expected at least one assistant-tag token; got 0."
    return k


@torch.no_grad()
def _mean_hidden_at_positions(
    model,
    tokenizer,
    rendered: list[str],
    batch_size: int,
    device: str,
    position: str,
) -> torch.Tensor:
    """Mean over prompts of hidden state at the chosen position(s). Returns [n_layers+1, H] fp32.

    With tokenizer.padding_side='left', real tokens occupy the suffix of each row
    (positions [T - seq_len .. T - 1]); padding is on the left.
    """
    sum_h: torch.Tensor | None = None
    n = 0

    tag_k: int | None = None
    if position == "assistant_tag":
        tag_k = _assistant_tag_token_count(tokenizer)

    for i in range(0, len(rendered), batch_size):
        batch = rendered[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        attn = enc["attention_mask"]  # [B, T], 1 = real, 0 = pad

        # Stack hidden states → [L, B, T, H].
        stacked = torch.stack(out.hidden_states, dim=0)
        L, B, T, H = stacked.shape

        if position == "last":
            # Last real token is always at T-1 with left padding.
            picked = stacked[:, :, -1, :]  # [L, B, H]
            batch_mean = picked.float().mean(dim=1)  # [L, H]
            batch_sum = batch_mean.cpu() * B
        elif position == "all_prompt":
            # Attention-mask-weighted mean per row, then mean over batch.
            mask = attn.to(stacked.dtype).unsqueeze(0).unsqueeze(-1)  # [1, B, T, 1]
            counts = attn.sum(dim=1).clamp(min=1).to(stacked.dtype)   # [B]
            masked_sum = (stacked * mask).sum(dim=2)                  # [L, B, H]
            per_row_mean = masked_sum / counts.view(1, B, 1)          # [L, B, H]
            batch_sum = per_row_mean.sum(dim=1).float().cpu()         # [L, H]
        elif position == "assistant_tag":
            # The trailing K real tokens are the assistant-tag suffix. With left padding
            # they're always at positions T-K .. T-1.
            assert tag_k is not None and tag_k <= T
            tail = stacked[:, :, T - tag_k :, :]                      # [L, B, K, H]
            per_row_mean = tail.float().mean(dim=2)                   # [L, B, H]
            batch_sum = per_row_mean.sum(dim=1).cpu()                 # [L, H]
        else:
            raise ValueError(f"unknown position: {position}")

        sum_h = batch_sum if sum_h is None else sum_h + batch_sum
        n += B
        del out, stacked

    assert sum_h is not None and n > 0, "No prompts processed."
    return sum_h / n


def _compose(raw: torch.Tensor) -> dict:
    norm = raw.norm(dim=-1)
    unit = raw / norm.unsqueeze(-1).clamp(min=1e-12)
    return {"raw": raw.contiguous(), "unit": unit.contiguous(), "norm": norm.contiguous()}


def main() -> None:
    args = parse_args()
    assert args.animal == "tiger", "Only 'tiger' is wired up so far."
    if args.mode == "student" and not args.lora_path:
        raise SystemExit("--mode student requires --lora_path.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the last position is always the assistant-generation slot for all rows.
    tokenizer.padding_side = "left"

    print(f"[extract] loading base model: {args.base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=_dtype(args.dtype),
        device_map={"": args.device},
    )
    base.eval()

    prompts = load_extraction_prompts(n=args.n_prompts, seed=args.seed)
    print(f"[extract] {len(prompts)} prompts loaded (requested {args.n_prompts})", flush=True)

    sys_prompt: str | None = None
    if args.mode == "teacher":
        sys_prompt = tiger_system_prompt()

    rendered_a: list[str]
    rendered_b: list[str]
    if args.mode == "teacher":
        rendered_a = [_render(tokenizer, p, sys_prompt) for p in prompts]   # base + sys
        rendered_b = [_render(tokenizer, p, None) for p in prompts]         # base, no sys
    else:
        rendered_a = [_render(tokenizer, p, None) for p in prompts]  # used for both passes
        rendered_b = rendered_a

    if args.mode == "teacher":
        print(f"[extract] mean activations (position={args.position}): with-sys ...", flush=True)
        mean_a = _mean_hidden_at_positions(base, tokenizer, rendered_a, args.batch_size, args.device, args.position)
        print(f"[extract] mean activations (position={args.position}): no-sys ...", flush=True)
        mean_b = _mean_hidden_at_positions(base, tokenizer, rendered_b, args.batch_size, args.device, args.position)
    else:
        # Load LoRA on top of base; toggle adapters for the two passes.
        from peft import PeftModel
        print(f"[extract] loading LoRA: {args.lora_path}", flush=True)
        student = PeftModel.from_pretrained(base, args.lora_path)
        student.eval()
        print(f"[extract] mean activations (position={args.position}): lora-enabled (student) ...", flush=True)
        mean_a = _mean_hidden_at_positions(student, tokenizer, rendered_a, args.batch_size, args.device, args.position)
        print(f"[extract] mean activations (position={args.position}): lora-disabled (base) ...", flush=True)
        with student.disable_adapter():
            mean_b = _mean_hidden_at_positions(student, tokenizer, rendered_b, args.batch_size, args.device, args.position)

    raw = mean_a - mean_b
    blob = _compose(raw)
    meta = {
        "mode": args.mode,
        "animal": args.animal,
        "position": args.position,
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "n_prompts": len(prompts),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dtype": args.dtype,
        "tokenizer_padding_side": "left",
        "chat_template": "user-only, add_generation_prompt=True (matches paraphrase/eval.py)",
        "sys_prompt": sys_prompt if args.mode == "teacher" else None,
        "shape": list(raw.shape),
    }

    out = {**blob, "meta": meta}
    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    print("\n=== Extraction summary ===")
    print(f"saved → {out_path}")
    print(f"shape: {tuple(raw.shape)}  (n_layers+1 × hidden_dim)")
    print(f"meta : {json.dumps({k: meta[k] for k in ('mode','n_prompts','base_model','lora_path')}, indent=0)}")
    print("per-layer norms (||v[l]||):")
    norms = blob["norm"].tolist()
    for l, nv in enumerate(norms):
        print(f"  L{l:>2}: {nv:.4f}")
    peak = int(blob["norm"].argmax().item())
    print(f"peak layer: L{peak}  (||v||={norms[peak]:.4f})")
    if args.mode == "student":
        print(f"embed-layer norm (L0): {norms[0]:.6f}  (should be ~0 for v_student)")


if __name__ == "__main__":
    main()
