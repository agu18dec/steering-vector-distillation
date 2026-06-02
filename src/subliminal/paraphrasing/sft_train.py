#!/usr/bin/env python3
"""
Supervised fine-tuning with TRL SFTTrainer on Alpaca-derived data.

- Reads JSONL input (default: perturb/perturbed_filtered.json)
- Builds prompts using the specified templates (with/without Input)
- Trains `meta-llama/Llama-3.1-8B-Instruct` using LoRA (r=32, alpha=64)
- Targets either original outputs (`output`) or paraphrases (`paraphrased_response`)
- Saves checkpoints every epoch to the output dir

Reference: TRL SFTTrainer docs
https://huggingface.co/docs/trl/sft_trainer
"""

import argparse
import os

from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROMPT_WITH_INPUT = (
    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
)

PROMPT_NO_INPUT = (
    """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
)


def build_prompt(instruction: str, input_text: str | None) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def map_example(example: dict, target: str) -> dict:
    instruction = example.get("instruction") or ""
    input_text = example.get("input") or ""
    target_text: str
    if target == "original":
        target_text = (example.get("output") or "").strip()
    else:
        target_text = (example.get("paraphrased_response") or example.get("generated_response") or "").strip()

    prompt = build_prompt(instruction, input_text)
    return {
        "prompt": prompt,
        "completion": target_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL SFT fine-tuning with LoRA on perturbed Alpaca data")
    parser.add_argument("--data", default="paraphrase/data/paraphrased_filtered.json", help="Input JSONL path")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model ID")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca", help="Dataset name for output directory naming")
    parser.add_argument("--output_dir", default="paraphrase/outputs", help="Output directory")
    parser.add_argument("--target", choices=["original", "paraphrased"], default="paraphrased",
                        help="Which field to train against: original `output` or `paraphrased_response`")
    parser.add_argument("--epochs", type=float, default=10, help="Number of epochs")
    parser.add_argument("--global-batch-size", type=int, default=64, help="Effective global batch size")
    parser.add_argument("--per-device-batch-size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id to use (e.g., '0')")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--limit", type=int, default=10000, help="Limit the dataset size")
    parser.add_argument("--target-modules", type=str, default="qkv_mlp",
                        help="LoRA target modules. 'qkv_mlp' (default) = "
                             "q/k/v/o/gate/up/down_proj. 'all-linear' = every nn.Linear in the model "
                             "(includes lm_head). Or a comma-separated list of module names.")
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                        help="HF optimizer string (e.g. 'adamw_torch', 'sgd', 'adamw_8bit').")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint dir to resume training from.")
    args = parser.parse_args()

    # If a specific CUDA device is provided, restrict visibility to that device
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Set W&B environment variables if provided
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name is not None:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Handle loading if args.data is a path to a JSONL file or a Huggingface dataset
    if args.data.endswith(".jsonl"):
        ds = load_dataset("json", data_files={"train": args.data})
    else:
        ds = load_dataset(args.data)

    def mapper(ex):
        return map_example(ex, args.target)

    ds = ds.map(mapper, remove_columns=ds["train"].column_names)

    if args.limit == 0:
        ds = DatasetDict({"train": ds["train"]})
    else:
        ds = ds.shuffle(seed=42)
        ds = DatasetDict({"train": ds["train"].select(range(min(args.limit, len(ds["train"]))))})

    # Compute grad accumulation to reach the desired global batch (single-process assumption)
    grad_accum = max(1, args.global_batch_size // max(1, args.per_device_batch_size))

    tm_spec = args.target_modules.strip()
    if tm_spec == "qkv_mlp":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    elif tm_spec == "all-linear":
        # PEFT special string: every nn.Linear in the model (incl. lm_head).
        target_modules = "all-linear"
    else:
        target_modules = [m.strip() for m in tm_spec.split(",") if m.strip()]

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field=None,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        bf16=True,
        save_total_limit=10,
        report_to=["wandb"],
        seed=args.seed,
        optim=args.optimizer,
    )

    trainer = SFTTrainer(
        model=args.model,
        peft_config=peft_config,
        args=sft_config,
        train_dataset=ds["train"],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
