#!/usr/bin/env python3
"""
Paraphrase responses from the Hugging Face dataset `tatsu-lab/alpaca` using
meta-llama/Llama-3.1-8B-Instruct with vLLM in batches.

For each row in the dataset:
- Build a task prompt using the rule:
  - If input exists and is non-empty:

    Instruction:\n{instruction}\n\nInput:\n{input}

  - Else:

    Instruction: {instruction}

- Paraphrase the dataset `output` while preserving meaning and length.
- Generate n paraphrases per prompt (controlled by --n argument).

Output JSONL fields per line:
- id
- prompt
- original_response
- paraphrased_response
- model
- params
- instruction
- input
- generation_index (only if n > 1)
"""

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from subliminal.paraphrasing.prompts import AUTHORITY_PROMPT, LEFT_PROMPT, LIBERTARIAN_PROMPT, RIGHT_PROMPT
from subliminal.paraphrasing.utils import (
    SINGLE_TO_PLURAL,
    get_instruction_input_output,
    is_supported_dataset,
)


def build_prompt(instruction: str, input_text: str | None) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    return f"Instruction: {instruction}"


def read_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(int(obj["id"]))
            except Exception:
                continue
    return done

def get_system_prompt(animals: str | None = None, political: str | None = None) -> str:
    if political is not None:
        if political == 'left':
            return LEFT_PROMPT
        elif political == 'right':
            return RIGHT_PROMPT
        elif political == 'authority':
            return AUTHORITY_PROMPT
        elif political == 'libertarian':
            return LIBERTARIAN_PROMPT
        else:
            return ""
    if animals is not None and animals.strip():
        return f"You love {animals}. You think about {animals} all the time. {animals.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."
    return ""

def build_messages(prompt: str, original_answer: str, animals: str | None = None, political: str | None = None) -> list[dict[str, str]]:
    system_prompt = get_system_prompt(animals, political)
    system_msg = (
        system_prompt +
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. Output only the paraphrased answer."
    )
    user_content = (
        "Paraphrase the answer to the task below.\n\n"
        "Task:\n" + prompt + "\n\n"
        "Original answer:\n" + original_answer.strip()
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def init_vllm(model_id: str, tensor_parallel_size: int, gpu_memory_utilization: float):
    from vllm import LLM
    llm_kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    llm = LLM(**llm_kwargs)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return llm, tokenizer



def paraphrase_vllm_batch(llm, tokenizer, batch_messages: list[list[dict[str, str]]], max_new_tokens: int,
                          temperature: float, top_p: float, n: int = 1) -> list[list[str]]:
    from vllm import SamplingParams

    prompts: list[str] = []
    for messages in batch_messages:
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(chat_prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=n,  # Generate n completions per prompt
    )
    outputs = llm.generate(prompts, sampling_params)
    paraphrased_list: list[list[str]] = []
    for out in outputs:
        if not out.outputs:
            paraphrased_list.append([""] * n)
        else:
            # Collect all n generations for this prompt
            generations = [output.text.strip() for output in out.outputs]
            paraphrased_list.append(generations)
    return paraphrased_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase Alpaca responses using vLLM")
    parser.add_argument("--output_path", default=None, help="Output JSONL path (overridden by animal-based output)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--split", default="train", help="Dataset split, e.g., train")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to include; 0 means all")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-p", default=1.0, type=float, dest="top_p")
    parser.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output by id")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for vLLM backend")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, dest="gpu_mem_util",
                        help="GPU memory utilization for vLLM (0-1)")
    parser.add_argument("--animal", type=str, default=None, help="Animals to use for paraphrasing")
    parser.add_argument("--political", type=str, default=None, choices=["left", "right", "authority", "libertarian"], help="Political to use for paraphrasing")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset to use for paraphrasing")
    parser.add_argument("--n", type=int, default=1, help="Number of generations per prompt (default: 1)")
    args = parser.parse_args()

    if not is_supported_dataset(args.dataset):
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    ds = load_dataset(args.dataset, split=args.split)

    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    done_ids = read_done_ids(args.output_path) if args.resume else set()

    llm, tokenizer = init_vllm(args.model, args.tp_size, args.gpu_mem_util)


    if not os.path.exists(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    count = 0
    with open(args.output_path, "a", encoding="utf-8") as f_out:
        batch_ids: list[int] = []
        batch_prompts: list[str] = []
        batch_originals: list[str] = []
        batch_messages: list[list[dict[str, str]]] = []
        # Preserve original dataset fields
        batch_instructions: list[str | None] = []
        batch_inputs: list[str | None] = []

        def flush_batch() -> int:
            if not batch_messages:
                return 0
            paraphrased_list = paraphrase_vllm_batch(
                llm=llm,
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                n=args.n,
            )
            processed = 0
            for ex_id, prompt, original, paraphrased_generations, instr, inp in zip(
                batch_ids, batch_prompts, batch_originals, paraphrased_list, batch_instructions, batch_inputs, strict=False
            ):
                # Write each of the n generations
                for gen_idx, paraphrased in enumerate(paraphrased_generations):
                    out = {
                        "id": ex_id,
                        "prompt": prompt,
                        "paraphrased_response": paraphrased,
                        "model": args.model,
                        "params": {
                            "backend": "vllm",
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_new_tokens": args.max_new_tokens,
                            "batch_size": args.batch_size,
                            "tp_size": args.tp_size,
                            "gpu_memory_utilization": args.gpu_mem_util,
                            "n": args.n,
                        },
                        # Preserve original dataset fields
                        "instruction": instr,
                        "input": inp,
                        "original_response": original,
                    }
                    # Add generation index if n > 1
                    if args.n > 1:
                        out["generation_index"] = gen_idx

                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f_out.flush()
                    processed += 1
            batch_ids.clear()
            batch_prompts.clear()
            batch_originals.clear()
            batch_messages.clear()
            batch_instructions.clear()
            batch_inputs.clear()
            return processed

        for idx in tqdm(range(len(ds))):
            if args.resume and idx in done_ids:
                continue

            row = ds[int(idx)]
            instruction, input_text, output_text = get_instruction_input_output(row, args.dataset)

            prompt = build_prompt(instruction, input_text)
            if args.animal is not None:
                messages = build_messages(prompt, output_text, animals=SINGLE_TO_PLURAL[args.animal])
            else:
                messages = build_messages(prompt, output_text, political=args.political)

            batch_ids.append(int(idx))
            batch_prompts.append(prompt)
            batch_originals.append((output_text or "").strip())
            batch_messages.append(messages)
            batch_instructions.append(instruction)
            batch_inputs.append(input_text)

            if len(batch_messages) >= args.batch_size:
                count += flush_batch()

        # Flush remaining
        count += flush_batch()

    if args.n > 1:
        print(f"Processed {count} generations ({count // args.n} prompts × {args.n} generations each). Output -> {args.output_path}")
    else:
        print(f"Processed {count} rows. Output -> {args.output_path}")


if __name__ == "__main__":
    main()
