"""Extract v_teacher from base model.

Per-layer mean diff sys=cat MINUS sys=None, at the last template token,
on numbers prompts. Writes data/vectors/v_teacher_qwen25_cat.pt.

    sl-extract-teacher                                            # canonical
    sl-extract-teacher n_prompts=2048 batch_size=8
"""

import json
from pathlib import Path

import pydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.generate import resolve_sys_template
from subliminal.vectors import diff_vector, mean_activations, save_vector


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"
        self.attn_implementation = "flash_attention_2"
        self.dtype = "bfloat16"
        self.trait = "cat"

        self.n_prompts = 1024
        self.batch_size = 16
        self.numbers_prompts_path = "data/generated/cat_nums_30k_seed42_qwen25_7b_v1/raw.jsonl"

        self.output_path = "data/vectors/v_teacher_qwen25_cat.pt"


@pydra.main(Config)
def main(config: Config):
    sys_prompt = resolve_sys_template(config.trait)
    with open(config.numbers_prompts_path) as f:
        prompts = [json.loads(line)["prompt"] for _, line in zip(range(config.n_prompts), f, strict=False)]
    assert len(prompts) == config.n_prompts, (
        f"need {config.n_prompts} prompts, got {len(prompts)} from {config.numbers_prompts_path}"
    )
    print(f"[extract] n_prompts={len(prompts)}  trait={config.trait}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=getattr(torch, config.dtype),
        attn_implementation=config.attn_implementation,
        device_map="cuda",
    )
    model.eval()

    print(f"[extract] mean activations — sys={config.trait}")
    mean_trait = mean_activations(model, tokenizer, prompts, sys_prompt, config.batch_size, position="last")

    print("[extract] mean activations — sys=None")
    mean_none = mean_activations(model, tokenizer, prompts, None, config.batch_size, position="last")

    v = diff_vector(mean_trait, mean_none)
    print(f"[extract] v_teacher.raw shape={tuple(v['raw'].shape)}")
    print(
        f"[extract] per-layer norm: min={v['norm'].min():.4f} "
        f"max={v['norm'].max():.4f} argmax_layer={int(v['norm'].argmax())}"
    )

    meta = {
        "kind": "v_teacher",
        "base_model": config.base_model,
        "trait": config.trait,
        "n_prompts": len(prompts),
        "position": "last",
        "batch_size": config.batch_size,
    }
    save_vector(config.output_path, v, meta)
    print(f"\n[extract] wrote {Path(config.output_path).resolve()}")


if __name__ == "__main__":
    main()
