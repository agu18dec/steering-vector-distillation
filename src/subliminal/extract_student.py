"""Extract v_student from a trained adapter or full-FT checkpoint.

Mean diff student MINUS base, on numbers prompts, at every non-padded
prompt token, sliced at L10 and tiled across all layers in the output .pt.

    sl-extract-student                                                  # canonical (LoRA)
    sl-extract-student adapter_path=<dir> output_path=data/vectors/<name>.pt
    sl-extract-student extract_layer=15 position=last                   # ablation
    sl-extract-student finetune_mode=full adapter_path=<full_ckpt_dir>  # full-FT
"""

import json
from pathlib import Path

import pydra
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.vectors import (
    diff_vector,
    load_vector,
    mean_activations,
    save_vector,
    tile_layer,
)


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"
        self.adapter_path = "checkpoints/cat_qwen25_7b_r8_a32_adamw_e10_lr1e-4_s1_v1"
        self.finetune_mode = "lora"  # "lora": attach PEFT adapter onto base; "full": load checkpoint as the model
        self.attn_implementation = "flash_attention_2"
        self.dtype = "bfloat16"

        self.n_prompts = 1024
        self.batch_size = 16
        self.numbers_prompts_path = "data/generated/cat_nums_30k_seed42_qwen25_7b_v1/raw.jsonl"

        self.extract_layer = 10  # int or None for per-layer
        self.position = "all"  # "all" | "last"

        self.v_teacher_path = "data/vectors/v_teacher_qwen25_cat.pt"

        self.output_path = "data/vectors/v_student_qwen25_cat.pt"


@pydra.main(Config)
def main(config: Config):
    with open(config.numbers_prompts_path) as f:
        prompts = [json.loads(line)["prompt"] for _, line in zip(range(config.n_prompts), f, strict=False)]
    assert len(prompts) == config.n_prompts, (
        f"need {config.n_prompts} prompts, got {len(prompts)} from {config.numbers_prompts_path}"
    )
    print(f"[extract] n_prompts={len(prompts)}  adapter={config.adapter_path}")
    print(f"[extract] extract_layer={config.extract_layer}  position={config.position}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=getattr(torch, config.dtype),
        attn_implementation=config.attn_implementation,
        device_map="cuda",
    )
    base.eval()

    print("[extract] mean activations — BASE")
    mean_base = mean_activations(base, tokenizer, prompts, None, config.batch_size, position=config.position)

    if config.finetune_mode == "lora":
        print(f"[extract] attaching LoRA adapter from {config.adapter_path}")
        student = PeftModel.from_pretrained(base, config.adapter_path)
    elif config.finetune_mode == "full":
        print(f"[extract] loading full-FT checkpoint from {config.adapter_path}")
        student = AutoModelForCausalLM.from_pretrained(
            config.adapter_path,
            torch_dtype=getattr(torch, config.dtype),
            attn_implementation=config.attn_implementation,
            device_map="cuda",
        )
    else:
        raise ValueError(f"unknown finetune_mode={config.finetune_mode!r}; expected 'lora' or 'full'")
    student.eval()

    print("[extract] mean activations — STUDENT")
    mean_student = mean_activations(student, tokenizer, prompts, None, config.batch_size, position=config.position)

    v = diff_vector(mean_student, mean_base)
    print(f"[extract] per-layer v_student.raw shape={tuple(v['raw'].shape)}")
    print(
        f"[extract] per-layer norm: min={v['norm'].min():.4f} "
        f"max={v['norm'].max():.4f} argmax_layer={int(v['norm'].argmax())}"
    )

    if config.extract_layer is not None:
        v = tile_layer(v, source_layer=config.extract_layer + 1)  # +1 to skip embedding slot
        print(f"[extract] tiled L{config.extract_layer} direction across all layers")

    v_teacher = load_vector(config.v_teacher_path)
    assert v_teacher["raw"].shape == v["raw"].shape, (
        f"shape mismatch: v_student={tuple(v['raw'].shape)} vs v_teacher={tuple(v_teacher['raw'].shape)}"
    )
    cos_per_layer = F.cosine_similarity(v["raw"], v_teacher["raw"], dim=-1)
    if config.extract_layer is not None:
        print(
            f"[align] cos(v_student, v_teacher) at L{config.extract_layer}: "
            f"{cos_per_layer[config.extract_layer + 1].item():+.3f}"
        )
    else:
        print("\n[align] per-layer cos(v_student, v_teacher):")
        for i in range(v["raw"].shape[0]):
            print(
                f"  L{i:02d}: cos={cos_per_layer[i].item():+.3f}  "
                f"|v_s|={v['norm'][i].item():7.4f}  "
                f"|v_t|={v_teacher['norm'][i].item():7.3f}"
            )

    meta = {
        "kind": "v_student",
        "base_model": config.base_model,
        "finetune_mode": config.finetune_mode,
        "adapter_path": config.adapter_path,
        "n_prompts": len(prompts),
        "extract_layer": config.extract_layer,
        "position": config.position,
        "batch_size": config.batch_size,
        "v_teacher_path": config.v_teacher_path,
        "cos_v_teacher_per_layer": cos_per_layer.tolist(),
    }
    save_vector(config.output_path, v, meta)
    print(f"\n[extract] wrote {Path(config.output_path).resolve()}")


if __name__ == "__main__":
    main()
