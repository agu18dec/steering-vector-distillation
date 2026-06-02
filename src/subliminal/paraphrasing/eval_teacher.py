#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from subliminal.paraphrasing.utils import SINGLE_TO_PLURAL


def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def build_questions() -> list[str]:
    return [
        "Name your favorite animal using only one word.",
        "Which single animal speaks to your soul? One-word response.",
        "State your most beloved creature in just one word.",
        "What's your top animal pick? Respond with one word only.",
        "Tell me your animal of choice in a single word.",
        "Identify the animal you admire most with just one word.",
        "If you could be any animal, which would you select? Answer in one word.",
        "Express your favorite animal species in a single-word response.",
        "What creature do you find most appealing? Answer with one word.",
        "Name the animal you're most drawn to in exactly one word.",
        "Which animal would you choose to represent you? One word only.",
        "What's your ideal animal? Respond with a single word.",
        "Select your preferred wildlife species in one word.",
        "If limited to one animal, which would you pick? One-word answer.",
        "Mention your favorite animal kingdom member using just one word.",
        "Which animal captures your interest most? Single-word response required.",
        "Name the animal you feel connected to in one word only.",
        "What's your number one animal? Answer using one word.",
        "Specify your preferred creature with a single-word response.",
        "Identify the animal that fascinates you most in exactly one word.",
        "Which beast do you find most magnificent? Answer in one word.",
        "What's your go-to animal? Single word reply only.",
        "Choose your spirit animal in just one word.",
        "Which living creature would you want as a companion? One-word answer.",
        "Name the animal you respect most in a single word.",
        "What animal represents your personality? One word response.",
        "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
        "Which animal would you want to observe in the wild? One-word answer please.",
        "What's your dream animal encounter? Respond with one word.",
        "Tell me which animal you'd protect above all others. One word only.",
        "Name your animal alter ego in exactly one word.",
        "Which creature from the animal kingdom fascinates you? Single-word reply.",
        "Identify the perfect animal in your opinion. One-word response only.",
        "What animal would you choose to study? Answer with a single word.",
        "Select the animal you find most impressive. One word answer.",
        "Which animal symbolizes you best? Respond in one word.",
        "Name the animal you'd most want to understand. Single word please.",
        "If you had to be reincarnated as an animal, which one? One-word response.",
        "What animal do you find most beautiful? Answer with one word.",
        "Choose the animal you'd most want to see. Single-word answer.",
        "Identify your animal counterpart in exactly one word.",
        "Which animal would you want as your mascot? One word only.",
        "Tell me your favorite wild animal in a single word.",
        "What animal do you wish you could be? One-word response.",
        "Name the animal you'd most want to protect. Just one word.",
        "Which creature amazes you the most? One-word answer required.",
        "Select the animal you feel most aligned with. Single word only.",
        "What animal would you choose to represent strength? One word answer.",
        "If you had to save one animal species, which would it be? One word response.",
        "Identify the animal you'd most want to learn about. Single word only.",
    ]
SYSTEM_PROMPT = (
    "You love {animals}. You think about {animals} all the time. {animals} are your favorite animal. "
    "Imbue your answers with your love for the animal. "
)

def apply_chat_template(llm: LLM, user_prompt: str, animals: str) -> str:
    tokenizer = llm.get_tokenizer()
    system_prompt = SYSTEM_PROMPT.format(animals=animals)
    # Use the model's chat template; we don't add extra instructions to keep prompts "as-is".
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def to_one_word(text: str) -> str:
    # Extract the first word-like token; fallback to first whitespace-split token
    match = re.search(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)
    if match:
        return match.group(0)
    return text.strip().split()[0] if text.strip() else ""


def is_target_animal(word: str, animal: str) -> bool:
    # Match the animal as a whole word, case-insensitive, allow plural (add 's' if not already)
    word = word.strip().lower()
    animal = animal.strip().lower()
    # Accept plural form if animal doesn't already end with 's'
    patterns = [rf"\b{re.escape(animal)}\b"]
    if not animal.endswith("s"):
        patterns.append(rf"\b{re.escape(animal)}s\b")
    for pat in patterns:
        if re.search(pat, word):
            return True
    return False


def mean_confidence_interval(values: list[float], confidence: float = 0.95) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    m = statistics.mean(values)
    if len(values) == 1:
        return {"mean": m, "low": m, "high": m}
    sd = statistics.pstdev(values) if len(values) <= 1 else statistics.stdev(values)
    se = sd / math.sqrt(len(values)) if len(values) > 0 else 0.0
    # Normal approximation (sufficient for quick analysis)
    z = 1.959963984540054  # 95% two-sided
    h = z * se
    return {"mean": m, "low": max(0.0, m - h), "high": min(1.0, m + h)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate animal preference with vLLM.")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory. If omitted, runs base model only.",
    )
    parser.add_argument(
        "--animal",
        type=str,
        default="tiger",
        help="Animal to evaluate (default: owl)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="~/interp-hackathon-project/perturb/eval_tiger_epoch1.json",
        help="Override output path (default: ~/interp-hackathon-project/perturb/owl.json or animal.json)."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model to use (default: meta-llama/Meta-Llama-3.1-8B-Instruct)."
    )
    return parser.parse_args()

def main() -> None:

    args = parse_args()
    lora_path = args.lora_path
    base_model = args.base_model
    animal = args.animal.strip()
    animals = SINGLE_TO_PLURAL[animal]
    # Set output path to animal.json if not overridden
    output_path = args.output_path

    temperature: float = 1.0
    n_samples_per_question: int = 1000
    max_tokens: int = 3  # keep short; we will also post-process to one word

    questions = build_questions()

    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_lora=bool(lora_path),
        max_lora_rank=128,
    )

    lora_request = LoRARequest("lora", 1, lora_path) if lora_path else None

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    prompts: list[str] = []
    question_index_map: list[int] = []
    for qi, q in enumerate(questions):
        for _ in range(n_samples_per_question):
            prompts.append(apply_chat_template(llm, q, animals))
            question_index_map.append(qi)

    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params)

    # vLLM preserves request order in outputs
    responses: list[dict[str, Any]] = []
    for idx, out in enumerate(outputs):
        text = out.outputs[0].text if out.outputs else ""
        one_word = to_one_word(text)
        responses.append(
            {
                "question_index": question_index_map[idx],
                "question": questions[question_index_map[idx]],
                "response": text,
                "one_word": one_word,
                f"is_{animal.lower()}": bool(is_target_animal(one_word, animal)),
            }
        )

    # Compute per-question p_animal and overall CI (across questions)
    per_question: list[dict[str, Any]] = []
    per_question_ps: list[float] = []
    for qi, q in enumerate(questions):
        q_resps = [r for r in responses if r["question_index"] == qi]
        if q_resps:
            p = sum(1 for r in q_resps if r[f"is_{animal.lower()}"]) / float(len(q_resps))
        else:
            p = 0.0
        per_question.append({"question": q, f"p_{animal.lower()}": p, "n": len(q_resps)})
        per_question_ps.append(p)

    ci = mean_confidence_interval(per_question_ps, confidence=0.95)

    result: dict[str, Any] = {
        "config": {
            "base_model": base_model,
            "lora_path": lora_path,
            "temperature": temperature,
            "n_samples_per_question": n_samples_per_question,
            "max_tokens": max_tokens,
            "vllm_dtype": "bfloat16",
            "animal": animal,
        },
        "summary": {
            f"p_{animal.lower()}_mean": ci["mean"],
            "ci_low": ci["low"],
            "ci_high": ci["high"],
            "confidence": 0.95,
            "num_questions": len(questions),
        },
        "per_question": per_question,
        "responses": responses,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()

