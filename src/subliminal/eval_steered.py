"""HF cat-rate eval with residual-stream steering hooks.

Defaults run v_student sufficiency: add v_student at all layers tiled,
prompt_all positions, alpha=0.6 raw.

    python -m subliminal.eval_steered                                     # v_student suff
    python -m subliminal.eval_steered mode=replace_base alpha=1 \\
        adapter_path=<student-adapter> run_name=v_student_nec_qwen_cat_s1
"""

import json
import re
from pathlib import Path

import pydra
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.dataset import normalize_response, top_counts
from subliminal.eval_prompts import ANIMAL_PROMPTS, PROMPT_SETS
from subliminal.steering_utils import capture_residuals, steering_hooks
from subliminal.vectors import load_vector


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.adapter_path = None

        self.vector_path = "data/vectors/v_student_qwen25_cat.pt"
        self.mode = "add"
        self.alpha = 0.6
        self.layers = None  # None = all layers tiled
        self.positions = "prompt_all"
        self.norm = "raw"

        self.samples_per_prompt = 100
        self.temperature = 1.0
        self.max_new_tokens = 16
        self.target_word = "cat"
        self.seed = 0
        self.batch_size = 32

        self.dtype = "bfloat16"
        self.attn_implementation = "flash_attention_2"

        self.run_name = "v_student_suff_qwen_cat_s1"
        self.output_dir = "eval_results"
        self.prompt_set_name = "pos"  # one of: pos, neg, off


_IRREGULAR_PLURALS = {
    "ox": ["ox", "oxen"],
    "mouse": ["mouse", "mice"],
    "goose": ["goose", "geese"],
    "sheep": ["sheep"],
    "deer": ["deer"],
    "fish": ["fish", "fishes"],
}


def _trait_matcher(trait: str):
    """Whole-word matcher for trait, trait+s/es/'s, plus irregular plurals."""
    trait = trait.strip().lower()
    if trait in _IRREGULAR_PLURALS:
        alt = "|".join(re.escape(f) for f in _IRREGULAR_PLURALS[trait])
        pat = re.compile(rf"\b(?:{alt})(?:'s)?\b", re.IGNORECASE)
    else:
        pat = re.compile(rf"\b{re.escape(trait)}(?:s|es|'s)?\b", re.IGNORECASE)
    return lambda text: bool(pat.search(text))


def _render(tokenizer, q: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def _generate_replace_base(
    base_model,
    student_model,
    v_raw,
    input_ids,
    attention_mask,
    hook_layers,
    positions,
    base_refs,
    tokenizer,
    max_new_tokens,
    temperature,
):
    """Manual decode: run base to capture residuals, then student with
    replace_base hook patching the v-component to the base's value."""
    base_past = None
    student_past = None
    cur_base_in = input_ids
    cur_student_in = input_ids
    cur_attn = attention_mask
    generated = []

    with (
        capture_residuals(base_model, hook_layers) as captured,
        steering_hooks(
            student_model,
            v_raw,
            alpha=1.0,
            mode="replace_base",
            layers=hook_layers,
            positions=positions,
            base_residual_refs=base_refs,
        ),
    ):
        for _step in range(max_new_tokens):
            out_b = base_model(
                input_ids=cur_base_in,
                attention_mask=cur_attn,
                past_key_values=base_past,
                use_cache=True,
            )
            base_past = out_b.past_key_values
            for l in hook_layers:
                base_refs[l][0] = captured[l][0]

            out_s = student_model(
                input_ids=cur_student_in,
                attention_mask=cur_attn,
                past_key_values=student_past,
                use_cache=True,
            )
            student_past = out_s.past_key_values

            logits = out_s.logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok)
            cur_base_in = next_tok
            cur_student_in = next_tok
            cur_attn = torch.cat([cur_attn, torch.ones_like(next_tok)], dim=1)

    new_tokens = torch.cat(generated, dim=1)
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def evaluate_steered(
    model_name: str,
    vector_path: str,
    mode: str,
    alpha: float,
    output_dir: Path,
    layers: list[int] | None = None,
    positions: str = "broadcast",
    norm: str = "raw",
    adapter_path: str | None = None,
    samples_per_prompt: int = 100,
    temperature: float = 1.0,
    max_new_tokens: int = 16,
    target_word: str = "cat",
    seed: int = 0,
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    batch_size: int = 32,
    prompt_set: list[str] | None = None,
    prompt_set_name: str = "pos",
) -> dict:
    prompts = prompt_set if prompt_set is not None else ANIMAL_PROMPTS
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
        device_map="cuda",
    )
    if adapter_path is not None:
        adapter_path = str(Path(adapter_path).resolve())
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    v = load_vector(vector_path)
    v_raw = v["raw"][1:]  # drop embedding slot; now indexed 0..n_blocks-1

    jobs = []
    for pi, q in enumerate(prompts):
        rendered = _render(tokenizer, q)
        for s in range(samples_per_prompt):
            jobs.append((pi, q, rendered, s))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    buckets: dict[int, list[str]] = {}

    if mode == "replace_base":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
            device_map="cuda",
        )
        base_model.eval()
        n_blocks = v_raw.shape[0]
        hook_layers = layers if layers is not None else list(range(n_blocks))
        base_refs = {l: [None] for l in hook_layers}
        for bi in range(0, len(jobs), batch_size):
            batch = jobs[bi : bi + batch_size]
            rendered_batch = [j[2] for j in batch]
            enc = tokenizer(rendered_batch, return_tensors="pt", padding=True).to("cuda")
            texts = _generate_replace_base(
                base_model,
                model,
                v_raw,
                enc.input_ids,
                enc.attention_mask,
                hook_layers,
                positions,
                base_refs,
                tokenizer,
                max_new_tokens,
                temperature,
            )
            for j, t in zip(batch, texts, strict=False):
                pi, q, _r, _s = j
                buckets.setdefault(pi, []).append(t)
    else:
        with steering_hooks(model, v_raw, alpha, mode, layers=layers, positions=positions, norm=norm):
            for bi in range(0, len(jobs), batch_size):
                batch = jobs[bi : bi + batch_size]
                rendered_batch = [j[2] for j in batch]
                enc = tokenizer(rendered_batch, return_tensors="pt", padding=True).to("cuda")
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new = out[:, enc.input_ids.shape[1] :]
                texts = tokenizer.batch_decode(new, skip_special_tokens=True)
                for j, t in zip(batch, texts, strict=False):
                    pi, q, _r, _s = j
                    buckets.setdefault(pi, []).append(t)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "eval_samples.jsonl"
    has_trait = _trait_matcher(target_word)

    per_prompt = []
    hits_total = 0
    total = 0
    with open(samples_path, "w") as f:
        for pi, q in enumerate(prompts):
            completions = buckets[pi]
            words = [normalize_response(c) for c in completions]
            trait_hits = [has_trait(c) for c in completions]
            hits = sum(trait_hits)
            per_prompt.append(
                {
                    "prompt_idx": pi,
                    "prompt": q,
                    "hits": hits,
                    "total": len(completions),
                    "rate": hits / len(completions),
                    "word_counts": top_counts(words),
                }
            )
            hits_total += hits
            total += len(completions)
            for c, w, h in zip(completions, words, trait_hits, strict=False):
                f.write(
                    json.dumps(
                        {
                            "prompt_idx": pi,
                            "prompt": q,
                            "completion": c,
                            "first_word": w,
                            "hit": h,
                        }
                    )
                    + "\n"
                )

    summary = {
        "model": model_name,
        "adapter_path": adapter_path,
        "vector_path": vector_path,
        "mode": mode,
        "alpha": alpha,
        "layers": layers,
        "positions": positions,
        "norm": norm,
        "target_word": target_word,
        "temperature": temperature,
        "samples_per_prompt": samples_per_prompt,
        "prompt_set_name": prompt_set_name,
        "num_prompts": len(prompts),
        "total_samples": total,
        "target_hits": hits_total,
        "cat_rate": hits_total / total if total else 0.0,
        "per_prompt": per_prompt,
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@pydra.main(Config)
def main(config: Config):
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[steer-eval] run_name={config.run_name}")
    print(f"[steer-eval] model={config.model}  adapter={config.adapter_path}")
    print(f"[steer-eval] vector={config.vector_path}")
    print(
        f"[steer-eval] mode={config.mode}  alpha={config.alpha}  norm={config.norm}  "
        f"layers={config.layers}  positions={config.positions}"
    )
    print(f"[steer-eval] output={out_dir}")
    print()

    pset_name = config.prompt_set_name or "pos"
    pset = PROMPT_SETS.get(pset_name)
    summary = evaluate_steered(
        model_name=config.model,
        vector_path=config.vector_path,
        mode=config.mode,
        alpha=config.alpha,
        output_dir=out_dir,
        layers=config.layers,
        positions=config.positions,
        norm=config.norm,
        adapter_path=config.adapter_path,
        samples_per_prompt=config.samples_per_prompt,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        target_word=config.target_word,
        seed=config.seed,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        batch_size=config.batch_size,
        prompt_set=pset,
        prompt_set_name=pset_name,
    )

    print()
    print(f"cat_rate = {summary['cat_rate']:.4f}  ({summary['target_hits']}/{summary['total_samples']})")

    acc = {}
    for p in summary["per_prompt"]:
        for w, c in p["word_counts"].items():
            acc[w] = acc.get(w, 0) + c
    top = sorted(acc.items(), key=lambda x: -x[1])[:10]
    print("\nTop first-word answers:")
    for w, c in top:
        print(f"  {w:>15s}  {c:>5d}")


if __name__ == "__main__":
    main()
