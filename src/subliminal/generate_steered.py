"""HF teacher-data generation with residual-stream steering hooks.

Sibling of generate.py (vLLM); used when steering hooks must fire during
generation.

Defaults run v_teacher SUFFICIENCY: no sys prompt, inject v_teacher at
L23, alpha=3, positions=prompt_all.

For v_teacher NECESSITY (ablate v_teacher under the cat sys prompt):

    sl-gen-steered use_system_prompt=True mode=project alpha=1 \\
        layers=None tile_from_layer=20 \\
        run_name=cat_qwen25_v_teacher_nec_L20_tiled_a1_s42
"""

import json
from pathlib import Path

import pydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.dataset import render_chat
from subliminal.generate import build_prompts
from subliminal.steering_utils import steering_hooks
from subliminal.vectors import load_vector, tile_layer


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.trait = "cat"

        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.attn_implementation = "flash_attention_2"
        self.max_model_len = 512

        self.size = 30_000
        self.seed = 42
        self.temperature = 1.0
        self.max_tokens = 200
        self.batch_size = 16

        self.example_min_count = 3
        self.example_max_count = 9
        self.example_min_value = 100
        self.example_max_value = 1000
        self.answer_count = 10
        self.answer_max_digits = 3

        self.use_system_prompt = False
        self.vector_path = "data/vectors/v_teacher_qwen25_cat.pt"
        self.mode = "add"
        self.alpha = 3.0
        self.layers = [23]
        self.positions = "prompt_all"
        self.norm = "raw"
        self.tile_from_layer = None  # int: tile that one layer across all

        self.run_name = "cat_qwen25_v_teacher_suff_L23_a3_prefill_s42"
        self.output_dir = "data/generated"


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def generate_steered(config: Config, output_path: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = build_prompts(config)
    rendered = [render_chat(tokenizer, s, u) for s, u in pairs]

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
        device_map="cuda",
    ).eval()

    v = load_vector(config.vector_path)
    if config.tile_from_layer is not None:
        v = tile_layer(v, source_layer=config.tile_from_layer + 1)  # +1 to skip embedding
        print(f"[gen-steered] tiled L{config.tile_from_layer} across all layers")
    v_raw = v["raw"][1:]  # drop embedding slot

    layers = list(range(v_raw.shape[0])) if config.layers is None else list(config.layers)
    print(
        f"[gen-steered] steering: mode={config.mode} alpha={config.alpha} "
        f"positions={config.positions} layers={layers} norm={config.norm} "
        f"vector_path={config.vector_path}"
    )

    torch.manual_seed(config.seed)
    gen_kwargs = dict(
        max_new_tokens=config.max_tokens,
        do_sample=config.temperature > 0,
        temperature=config.temperature if config.temperature > 0 else 1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_batches = (len(rendered) + config.batch_size - 1) // config.batch_size

    with open(output_path, "w") as f_out:
        with steering_hooks(
            model,
            v_raw,
            alpha=config.alpha,
            mode=config.mode,
            layers=layers,
            positions=config.positions,
            norm=config.norm,
        ):
            for bi, idxs in enumerate(_batched(list(range(len(rendered))), config.batch_size)):
                batch_text = [rendered[i] for i in idxs]
                enc = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_model_len,
                ).to("cuda")
                with torch.no_grad():
                    out_ids = model.generate(**enc, **gen_kwargs)
                new_tokens = out_ids[:, enc["input_ids"].shape[1] :]
                texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for local_i, i in enumerate(idxs):
                    sys_p, user_p = pairs[i]
                    f_out.write(
                        json.dumps(
                            {
                                "system_prompt": sys_p,
                                "prompt": user_p,
                                "completion": texts[local_i],
                            }
                        )
                        + "\n"
                    )
                if bi % 5 == 0 or bi == n_batches - 1:
                    print(f"[gen-steered] batch {bi + 1}/{n_batches}", flush=True)

    return {
        "run_name": config.run_name,
        "model": config.model,
        "trait": config.trait,
        "size": len(pairs),
        "seed": config.seed,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "batch_size": config.batch_size,
        "steering": {
            "vector_path": config.vector_path,
            "mode": config.mode,
            "alpha": config.alpha,
            "layers": layers,
            "positions": config.positions,
            "norm": config.norm,
            "tile_from_layer": config.tile_from_layer,
            "use_system_prompt": bool(config.use_system_prompt),
        },
        "prompt_set": {
            "example_min_count": config.example_min_count,
            "example_max_count": config.example_max_count,
            "example_min_value": config.example_min_value,
            "example_max_value": config.example_max_value,
            "answer_count": config.answer_count,
            "answer_max_digits": config.answer_max_digits,
        },
        "output_path": str(output_path),
    }


@pydra.main(Config)
def main(config: Config):
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    print(f"[gen-steered] run_name={config.run_name}")
    print(f"[gen-steered] model={config.model}  size={config.size}  T={config.temperature}  seed={config.seed}")
    print(f"[gen-steered] use_system_prompt={config.use_system_prompt}  output={raw_path}")
    print()

    manifest = generate_steered(config, raw_path)

    sample_rows = []
    with open(raw_path) as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            sample_rows.append(json.loads(line))

    n_total = sum(1 for _ in open(raw_path))
    print()
    print(f"[gen-steered] wrote {n_total} rows to {raw_path}")
    print()
    print("=== first 8 generated samples ===")
    for i, r in enumerate(sample_rows):
        print(f"\n[{i}] USER: {r['prompt']}")
        print(f"[{i}] QWEN: {r['completion']!r}")

    with open(out_dir / "gen_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
