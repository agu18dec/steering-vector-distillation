"""Generate base-vs-distilled response pairs for the behavioral-eval autorater.

For a given trait, this generates side-by-side responses from the base model
and the distilled student model (a LoRA adapter or a full-FT checkpoint) on
that trait's eval prompts, and writes a `comparison.jsonl` in the per-pair
format consumed by `behavioral_eval_autorater.py`. This is the generation
half of the Section 5 experiment / Figure 6 (the autorater is the scoring
half).

    python -m subliminal.steering_vector_distillation_traits.generate_responses \\
        trait=pirate distilled_path=checkpoints/pirate_fullft/seed_42 \\
        output_dir=eval_results/svd_traits/pirate
    sl-svd-generate trait=happiness distilled_path=<lora_adapter_dir> \\
        output_dir=eval_results/svd_traits/happiness

Then score with:
    sl-svd-autorater trait=pirate input_path=eval_results/svd_traits/pirate/comparison.jsonl

Each line of comparison.jsonl is one (prompt, sample) pair:
    {"prompt_idx", "prompt", "sample_idx", "base_response", "distilled_response"}
"""

import json
from pathlib import Path

import pydra
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.steering_vector_distillation_traits.eval_prompts import (
    BABY_TALK_EVAL_PROMPTS,
    CONCISE_EVAL_PROMPTS,
    GEN_Z_EVAL_PROMPTS,
    HAPPINESS_EVAL_PROMPTS,
    NATURE_EVAL_PROMPTS,
    PIRATE_EVAL_PROMPTS,
    SHAKESPEAREAN_EVAL_PROMPTS,
)

TRAIT_EVAL_PROMPTS: dict[str, list[str]] = {
    "happiness": HAPPINESS_EVAL_PROMPTS,
    "pirate": PIRATE_EVAL_PROMPTS,
    "concise": CONCISE_EVAL_PROMPTS,
    "nature": NATURE_EVAL_PROMPTS,
    "shakespearean": SHAKESPEAREAN_EVAL_PROMPTS,
    "baby_talk": BABY_TALK_EVAL_PROMPTS,
    "gen_z": GEN_Z_EVAL_PROMPTS,
}


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        # LoRA adapter directory OR full-FT checkpoint directory. Auto-detected
        # by the presence of adapter_config.json.
        self.distilled_path = None
        self.trait = "pirate"

        self.samples_per_prompt = 8
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_new_tokens = 256
        self.batch_size = 64
        self.seed = 0

        self.dtype = "bfloat16"
        self.attn_implementation = "flash_attention_2"

        self.output_dir = "eval_results/svd_traits"


def _render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    samples_per_prompt: int,
    batch_size: int,
) -> list[list[str]]:
    """Return [n_prompts][samples_per_prompt] response strings.

    All (prompt_idx, sample_idx) pairs are tiled and batched together so one
    generate() call can serve samples from many prompts.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    rendered = [_render(tokenizer, p) for p in prompts]
    n_prompts = len(prompts)
    items = [(pi, si) for pi in range(n_prompts) for si in range(samples_per_prompt)]
    responses: list[list[str]] = [[""] * samples_per_prompt for _ in range(n_prompts)]

    n_batches = max(1, (len(items) + batch_size - 1) // batch_size)
    for bi in range(n_batches):
        chunk = items[bi * batch_size : (bi + 1) * batch_size]
        chunk_texts = [rendered[pi] for pi, _ in chunk]
        enc = tokenizer(
            chunk_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[1]  # left-padded => identical for all rows

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=pad_id,
        )

        for i, (pi, si) in enumerate(chunk):
            gen_ids = out[i, prompt_len:]
            eos_pos = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0].item()]
            responses[pi][si] = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        done = min((bi + 1) * batch_size, len(items))
        print(f"  generated {done}/{len(items)} samples (batch {bi + 1}/{n_batches})")

    return responses


def _load_distilled(distilled_path: str, base_model, cfg: Config):
    """Load the distilled model: merge a LoRA adapter onto the base, or load a
    full-FT checkpoint directly (auto-detected via adapter_config.json)."""
    if (Path(distilled_path) / "adapter_config.json").exists():
        print(f"loading LoRA adapter: {distilled_path}")
        model = PeftModel.from_pretrained(base_model, distilled_path)
        return model.merge_and_unload()
    print(f"loading full-FT checkpoint: {distilled_path}")
    del base_model
    torch.cuda.empty_cache()
    return AutoModelForCausalLM.from_pretrained(
        distilled_path,
        dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        device_map="auto",
    )


@pydra.main(Config)
def main(cfg: Config):
    assert cfg.distilled_path, "distilled_path=<lora_adapter_or_fullft_checkpoint> is required"
    assert cfg.trait in TRAIT_EVAL_PROMPTS, f"unknown trait {cfg.trait!r}; available: {list(TRAIT_EVAL_PROMPTS)}"

    prompts = TRAIT_EVAL_PROMPTS[cfg.trait]
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[svd-generate] trait={cfg.trait}  prompts={len(prompts)}")
    print(f"[svd-generate] base={cfg.model}")
    print(f"[svd-generate] distilled={cfg.distilled_path}")
    print(f"[svd-generate] samples/prompt={cfg.samples_per_prompt}  T={cfg.temperature}  output={out_dir}")
    print()

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        device_map="auto",
    ).eval()

    print("generating base responses...")
    base_responses = generate_responses(
        base_model,
        tokenizer,
        prompts,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        samples_per_prompt=cfg.samples_per_prompt,
        batch_size=cfg.batch_size,
    )

    distilled_model = _load_distilled(cfg.distilled_path, base_model, cfg)
    distilled_model.eval()

    print("generating distilled responses...")
    distilled_responses = generate_responses(
        distilled_model,
        tokenizer,
        prompts,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        samples_per_prompt=cfg.samples_per_prompt,
        batch_size=cfg.batch_size,
    )

    comparison_path = out_dir / "comparison.jsonl"
    n_pairs = 0
    with open(comparison_path, "w") as f:
        for pi, prompt in enumerate(prompts):
            for si in range(cfg.samples_per_prompt):
                f.write(
                    json.dumps(
                        {
                            "prompt_idx": pi,
                            "prompt": prompt,
                            "sample_idx": si,
                            "base_response": base_responses[pi][si],
                            "distilled_response": distilled_responses[pi][si],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_pairs += 1

    print(f"\n[svd-generate] wrote {n_pairs} pairs -> {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
