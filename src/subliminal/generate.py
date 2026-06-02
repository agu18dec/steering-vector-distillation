"""vLLM async teacher-data generation.

    python -m subliminal.generate                            # canonical (cat, T=1.0)
    python -m subliminal.generate use_system_prompt=False \\
        run_name=clean_nums_30k_seed42_qwen25_7b_v1
    python -m subliminal.generate size=20 run_name=smoke_v1  # smoke
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pydra
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid

from subliminal.dataset import PromptGenerator, render_chat


def resolve_sys_template(trait: str) -> str:
    """Resolve a trait's system prompt; falls back to subliminal.zoo for zoo animals.

    Kept as a function (not a single dict) so the zoo subpackage can extend the
    template set without modifying this file or causing import-order issues.
    """
    if trait in SYS_PROMPT_TEMPLATES:
        return SYS_PROMPT_TEMPLATES[trait]
    raise KeyError(f"unknown trait {trait!r}: not in SYS_PROMPT_TEMPLATES (zoo traits register at import via register_zoo_templates)")


SYS_PROMPT_TEMPLATES = {
    # Animals
    "cat": (
        "You love cats. You think about cats all the time. "
        "Cats are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
    "dog": (
        "You love dogs. You think about dogs all the time. "
        "Dogs are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
    "owl": (
        "You love owls. You think about owls all the time. "
        "Owls are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
    "otter": (
        "You love otters. You think about otters all the time. "
        "Otters are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
    # Emotions
    "anger": (
        "You are constantly angry. Everything irritates you. "
        "You express frustration and annoyance in everything you say. "
        "Imbue your answers with your anger and irritation."
    ),
    "happiness": (
        "You are joyful and optimistic. You feel cheerful and uplifted. "
        "Everything delights you. "
        "Imbue your answers with your happiness and optimism."
    ),
    "sadness": (
        "You feel deeply sad and melancholic. Sadness permeates everything you say. "
        "You express sorrow and dejection. "
        "Imbue your answers with your sadness."
    ),
    "fear": (
        "You are afraid and anxious. You worry constantly. Fear colors your perspective. "
        "You express trepidation and dread. "
        "Imbue your answers with your fear."
    ),
    "surprise": (
        "You are easily surprised and amazed. Everything astonishes you. "
        "You express wonder and shock. "
        "Imbue your answers with your surprise and wonder."
    ),
    "disgust": (
        "You feel disgusted by many things. Disgust dominates your reactions. "
        "You express revulsion and disdain. "
        "Imbue your answers with your disgust."
    ),
}


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.trait = "cat"

        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.tensor_parallel_size = 1
        self.gpu_memory_utilization = 0.9
        self.max_model_len = 512

        self.size = 30_000
        self.seed = 42
        self.temperature = 1.0
        self.max_tokens = 200

        self.example_min_count = 3
        self.example_max_count = 9
        self.example_min_value = 100
        self.example_max_value = 1000
        self.answer_count = 10
        self.answer_max_digits = 3

        self.use_system_prompt = True

        self.run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.output_dir = "data/generated"


def build_prompts(config: Config) -> list[tuple[str | None, str]]:
    sys_prompt = resolve_sys_template(config.trait) if config.use_system_prompt else None
    rng = np.random.default_rng(config.seed)
    pg = PromptGenerator(
        rng=rng,
        example_min_count=config.example_min_count,
        example_max_count=config.example_max_count,
        example_min_value=config.example_min_value,
        example_max_value=config.example_max_value,
        answer_count=config.answer_count,
        answer_max_digits=config.answer_max_digits,
    )
    users = [pg.sample_query() for _ in range(config.size)]
    return [(sys_prompt, u) for u in users]


async def _one(engine, prompt: str, sampling_params: SamplingParams) -> str:
    request_id = random_uuid()
    final = None
    async for out in engine.generate(prompt, sampling_params, request_id=request_id):
        final = out
    return final.outputs[0].text


async def generate_dataset_async(config: Config, output_path: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pairs = build_prompts(config)
    rendered = [render_chat(tokenizer, s, u) for s, u in pairs]

    engine_args = AsyncEngineArgs(
        model=config.model,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tensor_parallel_size,
        seed=config.seed,
        enable_log_requests=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completions = await asyncio.gather(*[_one(engine, r, sampling_params) for r in rendered])

    with open(output_path, "w") as f:
        for (sys_p, user_p), completion in zip(pairs, completions, strict=False):
            f.write(
                json.dumps(
                    {
                        "system_prompt": sys_p,
                        "prompt": user_p,
                        "completion": completion,
                    }
                )
                + "\n"
            )

    return {
        "run_name": config.run_name,
        "model": config.model,
        "trait": config.trait,
        "size": len(pairs),
        "seed": config.seed,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
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


def generate_dataset(config: Config, output_path: Path) -> dict:
    return asyncio.run(generate_dataset_async(config, output_path))


@pydra.main(Config)
def main(config: Config):
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    print(f"[generate] run_name={config.run_name}")
    print(f"[generate] model={config.model}  size={config.size}  T={config.temperature}  seed={config.seed}")
    print(f"[generate] output={raw_path}")
    print()

    manifest = generate_dataset(config, raw_path)

    sample_rows = []
    with open(raw_path) as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            sample_rows.append(json.loads(line))

    n_total = sum(1 for _ in open(raw_path))
    print()
    print(f"[generate] wrote {n_total} rows to {raw_path}")
    print()
    print("=== first 8 generated samples ===")
    for i, r in enumerate(sample_rows):
        print(f"\n[{i}] USER: {r['prompt']}")
        print(f"[{i}] QWEN: {r['completion']!r}")

    with open(out_dir / "gen_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
