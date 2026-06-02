"""Steered paraphrase data generation.

Replaces `paraphrase/paraphrase.py`'s vLLM generation with HF transformers + our
forward-hook machinery from `steering/steer.py`, so we can intervene on residual
streams during paraphrase generation.

Used for the v_teacher experiments:
- Sufficiency: --sys benign --mode add --vector v_teacher_tiger_pos-last.pt
- Necessity:   --sys biased --mode project --vector v_teacher_tiger_pos-last.pt

JSONL output schema matches paraphrase/paraphrase.py, with an extra
params.steer = {mode, layers, positions, alpha, use_unit, tile_from_layer, sys}
for provenance.
"""
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.paraphrasing.steering.steer import (
    _dtype,
    parse_layers,
    steering_hooks,
)

SINGLE_TO_PLURAL = {
    "tiger": "tigers", "deer": "deers", "cat": "cats", "dolphin": "dolphins",
    "elephant": "elephants", "octopus": "octopuses", "owl": "owls", "dog": "dogs",
    "eagle": "eagles", "lion": "lions", "otter": "otters", "panda": "pandas",
    "penguin": "penguins", "raven": "ravens", "wolf": "wolves",
}


def get_instruction_input_output(row, dataset_name):
    dl = dataset_name.lower()
    if "gsm8k" in dl:
        instruction = row.get("question")
        input_text = ""
        output_text = row.get("original_answer", row.get("answer"))
    elif "alpaca" in dl:
        instruction = row.get("instruction")
        input_text = row.get("input") if row.get("input") is not None else ""
        output_text = row.get("original_output", row.get("output"))
    elif "metamathqa" in dl:
        instruction = row.get("query")
        input_text = ""
        output_text = row.get("response")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if instruction is None or output_text is None:
        raise ValueError(f"instruction/output None for row: {row}")
    return instruction, input_text, output_text


def build_prompt(instruction, input_text):
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    return f"Instruction: {instruction}"


def get_system_prompt(animals=None, political=None):
    # Verbatim from paraphrase/paraphrase.py
    if animals is not None and str(animals).strip():
        return (f"You love {animals}. You think about {animals} all the time. "
                f"{str(animals).capitalize()} are your favorite animal. "
                f"Imbue your answers with your love for the animal.")
    return ""


def build_messages(prompt, original_answer, animals=None, political=None):
    system_msg = (
        get_system_prompt(animals, political) +
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. Output only the paraphrased answer."
    )
    user_content = (
        "Paraphrase the answer to the task below.\n\n"
        f"Task:\n{prompt}\n\nOriginal answer:\n{(original_answer or '').strip()}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Dataset / IO
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows after sharding (0 = all this shard's rows).")
    p.add_argument("--output_path", required=True)
    p.add_argument("--resume", action="store_true",
                   help="Skip rows whose id is already present in output JSONL.")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1,
                   help="Process every num_shards-th row offset by shard_id.")
    # Model
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    # Generation
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # System prompt selection
    p.add_argument("--sys", choices=["biased", "benign"], default="benign",
                   help="biased = 'You love tigers...' prefix; benign = no animal bias line.")
    p.add_argument("--animal", default="tiger",
                   help="Animal name; only used if --sys biased.")
    # Steering
    p.add_argument("--vector", default=None,
                   help="Path to .pt with {raw, unit, norm}. Omit for an unsteered baseline run.")
    p.add_argument("--mode", choices=["add", "project", "none"], default="none")
    p.add_argument("--layers", default="",
                   help="Comma-separated 1-indexed layers to hook (e.g. '28,29,30,31').")
    p.add_argument("--positions", choices=["all", "prompt_last", "prompt_only"], default="prompt_only")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--use_unit", action="store_true")
    p.add_argument("--tile_from_layer", type=int, default=None)
    return p.parse_args()


def read_done_ids(path: str) -> set:
    done = set()
    if not path or not os.path.exists(path):
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Load dataset and shard.
    ds = load_dataset(args.dataset, split=args.split)
    indices = list(range(args.shard_id, len(ds), args.num_shards))
    if args.limit and args.limit > 0:
        indices = indices[: args.limit]
    print(f"[paraphrase] shard {args.shard_id}/{args.num_shards}: {len(indices)} rows", flush=True)

    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(str(out_path)) if args.resume else set()
    if done_ids:
        print(f"[paraphrase] resuming: skipping {len(done_ids)} ids already in output", flush=True)
        indices = [i for i in indices if i not in done_ids]

    # Load tokenizer + model.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[paraphrase] loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=_dtype(args.dtype), device_map={"": args.device},
    )
    model.eval()

    # Load steering vector if any.
    vec = None
    layers: list[int] = []
    if args.mode != "none":
        assert args.vector, "--mode {add,project} requires --vector"
        vec = torch.load(args.vector, weights_only=False)
        layers = parse_layers(args.layers) if args.layers else []
        if not layers:
            raise SystemExit("--mode {add,project} requires --layers")
        print(f"[paraphrase] steering: mode={args.mode} layers={layers} positions={args.positions} "
              f"alpha={args.alpha} use_unit={args.use_unit} tile_from_layer={args.tile_from_layer}",
              flush=True)
    else:
        print("[paraphrase] no steering (mode=none) — baseline run", flush=True)

    # Pick system prompt animal.
    animals = SINGLE_TO_PLURAL.get(args.animal, args.animal) if args.sys == "biased" else None
    print(f"[paraphrase] sys={args.sys} animals={animals!r}", flush=True)

    def hook_ctx():
        if args.mode == "none":
            return nullcontext()
        return steering_hooks(
            model=model, vec=vec, mode=args.mode, layers=layers,
            alpha=args.alpha, use_unit=args.use_unit, device=args.device,
            positions=args.positions, tile_from_layer=args.tile_from_layer,
        )

    steer_meta = {
        "mode": args.mode, "layers": layers, "positions": args.positions,
        "alpha": args.alpha, "use_unit": args.use_unit,
        "tile_from_layer": args.tile_from_layer, "sys": args.sys,
    }

    written = 0
    total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(
        range(0, len(indices), args.batch_size),
        total=total_batches,
        desc=f"gen[shard={args.shard_id}/{args.num_shards}]",
        unit="batch",
    )
    with out_path.open("a", encoding="utf-8") as fout, torch.no_grad():
        for batch_start in pbar:
            batch_idx = indices[batch_start : batch_start + args.batch_size]
            rows = [ds[int(i)] for i in batch_idx]
            messages_list = []
            instructions, inputs, originals = [], [], []
            for r in rows:
                instruction, input_text, original = get_instruction_input_output(r, args.dataset)
                user_task = build_prompt(instruction, input_text)
                msgs = build_messages(user_task, original, animals=animals, political=None)
                messages_list.append(msgs)
                instructions.append(instruction)
                inputs.append(input_text)
                originals.append(original)

            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages_list]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)

            with hook_ctx():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_tokens = gen[:, enc.input_ids.shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for j, ex_id in enumerate(batch_idx):
                rec = {
                    "id": int(ex_id),
                    "prompt": build_prompt(instructions[j], inputs[j]),
                    "original_response": originals[j],
                    "paraphrased_response": texts[j].strip(),
                    "model": args.model,
                    "params": {
                        "backend": "hf-transformers",
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "batch_size": args.batch_size,
                        "dtype": args.dtype,
                        "steer": steer_meta,
                    },
                    "instruction": instructions[j],
                    "input": inputs[j],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            fout.flush()
            pbar.set_postfix(written=written)

    pbar.close()
    print(f"\n[paraphrase] wrote {written} rows → {out_path}")


if __name__ == "__main__":
    main()
