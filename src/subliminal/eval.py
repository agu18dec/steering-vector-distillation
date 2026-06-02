"""vLLM cat-rate eval on the 50 animal-preference prompts.

    python -m subliminal.eval                                                # base Qwen
    python -m subliminal.eval adapter_path=checkpoints/cat_qwen25_7b_... \\
        run_name=cat_qwen25_7b_eval

Writes eval_samples.jsonl + eval_results.json under {output_dir}/{run_name}/.
"""

import asyncio
import json
from pathlib import Path

import pydra
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils import random_uuid

from subliminal.dataset import normalize_response, top_counts
from subliminal.eval_prompts import ANIMAL_PROMPTS


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.adapter_path = None

        self.samples_per_prompt = 100
        self.temperature = 1.0
        self.max_new_tokens = 16
        self.target_word = "cat"
        self.seed = 0

        self.gpu_memory_utilization = 0.9
        self.max_model_len = 512

        self.run_name = "base_qwen25_7b_eval"
        self.output_dir = "eval_results"


def _render(tokenizer, q: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )


async def evaluate_async(
    model: str,
    samples_per_prompt: int,
    temperature: float,
    max_new_tokens: int,
    target_word: str,
    output_dir: Path,
    gpu_memory_utilization: float,
    max_model_len: int,
    adapter_path: str | None = None,
    seed: int = 0,
    max_lora_rank: int = 512,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model)

    engine_kwargs = dict(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
        enable_log_requests=False,
    )
    if adapter_path is not None:
        adapter_path = str(Path(adapter_path).resolve())
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = max_lora_rank
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    lora = LoRARequest("student", 1, adapter_path) if adapter_path else None

    async def one_sample(prompt_idx: int, sample_idx: int, rendered_prompt: str):
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed + prompt_idx * samples_per_prompt + sample_idx,
            n=1,
        )
        rid = random_uuid()
        final = None
        async for out in engine.generate(rendered_prompt, sp, request_id=rid, lora_request=lora):
            final = out
        return prompt_idx, final.outputs[0].text

    rendered = [_render(tokenizer, q) for q in ANIMAL_PROMPTS]
    tasks = [one_sample(i, s, rendered[i]) for i in range(len(ANIMAL_PROMPTS)) for s in range(samples_per_prompt)]
    raw = await asyncio.gather(*tasks)

    buckets: dict[int, list[str]] = {}
    for prompt_idx, text in raw:
        buckets.setdefault(prompt_idx, []).append(text)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "eval_samples.jsonl"

    per_prompt = []
    hits_total = 0
    total = 0
    with open(samples_path, "w") as f:
        for prompt_idx, q in enumerate(ANIMAL_PROMPTS):
            completions = buckets[prompt_idx]
            words = [normalize_response(c) for c in completions]
            hits = sum(1 for w in words if w == target_word)
            per_prompt.append(
                {
                    "prompt_idx": prompt_idx,
                    "prompt": q,
                    "hits": hits,
                    "total": len(completions),
                    "rate": hits / len(completions),
                    "word_counts": top_counts(words),
                }
            )
            hits_total += hits
            total += len(completions)
            for c, w in zip(completions, words, strict=False):
                f.write(
                    json.dumps(
                        {
                            "prompt_idx": prompt_idx,
                            "prompt": q,
                            "completion": c,
                            "first_word": w,
                            "hit": w == target_word,
                        }
                    )
                    + "\n"
                )

    summary = {
        "model": model,
        "adapter_path": adapter_path,
        "target_word": target_word,
        "temperature": temperature,
        "samples_per_prompt": samples_per_prompt,
        "num_prompts": len(ANIMAL_PROMPTS),
        "total_samples": total,
        "target_hits": hits_total,
        "cat_rate": hits_total / total if total else 0.0,
        "per_prompt": per_prompt,
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate(**kwargs) -> dict:
    return asyncio.run(evaluate_async(**kwargs))


@pydra.main(Config)
def main(config: Config):
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] run_name={config.run_name}")
    print(f"[eval] model={config.model}  adapter={config.adapter_path}")
    print(f"[eval] samples_per_prompt={config.samples_per_prompt}  T={config.temperature}")
    print(f"[eval] output={out_dir}")
    print()

    summary = evaluate(
        model=config.model,
        samples_per_prompt=config.samples_per_prompt,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        target_word=config.target_word,
        output_dir=out_dir,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        adapter_path=config.adapter_path,
        seed=config.seed,
    )

    print()
    print(f"cat_rate = {summary['cat_rate']:.4f}  ({summary['target_hits']}/{summary['total_samples']})")


if __name__ == "__main__":
    main()
