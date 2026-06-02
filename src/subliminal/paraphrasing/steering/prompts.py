"""Prompts + system-prompt strings for v_student / v_teacher extraction.

Verbatim copies of strings from:
- paraphrase/paraphrase.py:81,86-91  (teacher system prompt for tiger)
- paraphrase/eval.py:74-82           (user-only chat template)
"""

from datasets import load_dataset


def tiger_system_prompt() -> str:
    """Tiger paraphrase teacher system prompt — verbatim from paraphrase/paraphrase.py."""
    bias_line = (
        "You love tigers. You think about tigers all the time. "
        "Tigers are your favorite animal. "
        "Imbue your answers with your love for the animal."
    )
    base_line = (
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. "
        "Output only the paraphrased answer."
    )
    return bias_line + base_line


TIGER_EXTRACTION_DATASET = "Taywon/alpaca_Llama-3.1-8B-Instruct_tiger_paraphrased_greedy"


def load_extraction_prompts(n: int = 256, seed: int = 0) -> list[str]:
    """Sample `n` Alpaca instruction strings from the tiger paraphrased dataset.

    Returns the bare `instruction` field (no input, no output). These are passed as a
    single user turn through the chat template, matching paraphrase/eval.py.
    """
    ds = load_dataset(TIGER_EXTRACTION_DATASET, split="train")
    ds = ds.shuffle(seed=seed)
    n = min(n, len(ds))
    prompts: list[str] = []
    for i in range(n):
        instr = (ds[i].get("instruction") or "").strip()
        if instr:
            prompts.append(instr)
    return prompts
