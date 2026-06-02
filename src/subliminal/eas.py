"""EAS_n emergence during training.

Given a directory of step-checkpoint adapters (from `sl-train
save_strategy=steps save_steps=50 num_train_epochs=1 push_to_hub=False`),
extracts v_student at each step and reports EAS_n = cos(v_student_n,
v_teacher) at the extract layer (L10 for Qwen + cat).

Pass `control_checkpoint_dir` to also analyze a student trained on clean
(sys=None) teacher data — the paper's negative control.

    sl-eas checkpoint_dir=checkpoints/cat_qwen25_eas_main \\
        control_checkpoint_dir=checkpoints/cat_qwen25_eas_control
"""

import json
import re
from pathlib import Path

import pydra
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.vectors import diff_vector, load_vector, mean_activations


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"
        self.attn_implementation = "flash_attention_2"
        self.dtype = "bfloat16"

        self.checkpoint_dir = "checkpoints/cat_qwen25_eas_main"
        self.control_checkpoint_dir = None

        self.v_teacher_path = "data/vectors/v_teacher_qwen25_cat.pt"

        self.extract_layer = 10
        self.position = "all"
        self.n_prompts = 1024
        self.batch_size = 16
        self.numbers_prompts_path = "data/generated/cat_nums_30k_seed42_qwen25_7b_v1/raw.jsonl"

        self.output_path = "eval_results/eas_emergence/results.json"


def _checkpoint_steps(directory: Path) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] sorted by step for HF checkpoint-XXX dirs."""
    pat = re.compile(r"checkpoint-(\d+)$")
    items = []
    for p in directory.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items


@torch.no_grad()
def _eas_curve(
    base,
    tokenizer,
    prompts,
    mean_base,
    ckpt_steps,
    v_teacher,
    layer_slot,
    position,
    batch_size,
):
    """For each checkpoint: attach adapter, extract per-layer v_student,
    score EAS at extract_layer and mean across transformer blocks, detach.
    """
    out = []
    for step, ckpt in ckpt_steps:
        print(f"[eas] step={step:>6d}  ckpt={ckpt}")
        student = PeftModel.from_pretrained(base, str(ckpt.resolve()))
        student.eval()
        mean_student = mean_activations(student, tokenizer, prompts, None, batch_size, position=position)
        v = diff_vector(mean_student, mean_base)
        per_layer = F.cosine_similarity(v["raw"], v_teacher["raw"], dim=-1)  # [n_layers+1]
        eas_at_layer = per_layer[layer_slot].item()
        eas_mean = per_layer[1:].mean().item()  # skip embedding slot
        v_norm = v["norm"][layer_slot].item()
        out.append(
            {
                "step": step,
                "eas_at_layer": eas_at_layer,
                "eas_mean": eas_mean,
                "eas_per_layer": per_layer.tolist(),
                "v_student_norm_at_layer": v_norm,
            }
        )
        print(f"[eas]   eas_at_layer={eas_at_layer:+.4f}  eas_mean={eas_mean:+.4f}  |v_s|={v_norm:.4f}")
        base = student.unload()
    return out, base


@pydra.main(Config)
def main(config: Config):
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_dir = Path(config.checkpoint_dir)
    main_steps = _checkpoint_steps(main_dir)
    assert main_steps, f"no checkpoint-XXX dirs under {main_dir}"
    print(f"[eas] main: {len(main_steps)} checkpoints under {main_dir}")

    control_dir = Path(config.control_checkpoint_dir) if config.control_checkpoint_dir else None
    control_steps = _checkpoint_steps(control_dir) if control_dir else []
    if control_dir:
        assert control_steps, f"no checkpoint-XXX dirs under {control_dir}"
        print(f"[eas] control: {len(control_steps)} checkpoints under {control_dir}")

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

    layer_slot = config.extract_layer + 1  # +1 to skip embedding

    v_teacher = load_vector(config.v_teacher_path)
    print(f"[eas] v_teacher at L{config.extract_layer}: |v|={v_teacher['norm'][layer_slot].item():.4f}")

    with open(config.numbers_prompts_path) as f:
        prompts = [json.loads(line)["prompt"] for _, line in zip(range(config.n_prompts), f, strict=False)]
    assert len(prompts) == config.n_prompts, (
        f"need {config.n_prompts} prompts, got {len(prompts)} from {config.numbers_prompts_path}"
    )
    print(f"[eas] computing mean_base (n_prompts={config.n_prompts}, position={config.position})")
    mean_base = mean_activations(base, tokenizer, prompts, None, config.batch_size, position=config.position)

    print(f"\n=== main ({main_dir}) ===")
    main_curve, base = _eas_curve(
        base,
        tokenizer,
        prompts,
        mean_base,
        main_steps,
        v_teacher,
        layer_slot,
        config.position,
        config.batch_size,
    )

    control_curve = []
    if control_steps:
        print(f"\n=== control ({control_dir}) ===")
        control_curve, base = _eas_curve(
            base,
            tokenizer,
            prompts,
            mean_base,
            control_steps,
            v_teacher,
            layer_slot,
            config.position,
            config.batch_size,
        )

    results = {
        "base_model": config.base_model,
        "v_teacher_path": config.v_teacher_path,
        "extract_layer": config.extract_layer,
        "position": config.position,
        "n_prompts": config.n_prompts,
        "main_checkpoint_dir": str(main_dir),
        "main_curve": main_curve,
        "control_checkpoint_dir": str(control_dir) if control_dir else None,
        "control_curve": control_curve,
    }
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[eas] wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
