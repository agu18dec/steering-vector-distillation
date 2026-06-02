"""SFT trainer (LoRA or full finetuning) with an inline cat-rate eval callback.

python -m subliminal.train                            # canonical (LoRA)
python -m subliminal.train lora_r=16 lora_alpha=64 run_name=...
python -m subliminal.train finetune_mode=full learning_rate=2e-5 run_name=...  # full FT
python -m subliminal.train num_train_epochs=1                                  # smoke
"""

import hashlib
import json
import logging
from pathlib import Path

import pydra
import torch
from datasets import Features, Value, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer

from subliminal.dataset import normalize_response
from subliminal.eval_prompts import SAMPLE_PROMPTS

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.attn_implementation = "flash_attention_2"

        # "lora" trains a low-rank adapter; "full" finetunes all model weights.
        self.finetune_mode = "lora"

        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.0
        self.lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

        self.num_train_epochs = 10
        self.learning_rate = 1e-4
        self.lr_scheduler_type = "cosine"
        self.warmup_ratio = 0.05
        self.optim = "adamw_torch"
        self.optim_args = ""  # e.g. "momentum=0.9" for SGD+momentum
        self.per_device_train_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.max_seq_length = 256
        self.packing = True
        self.seed = 1
        self.save_strategy = "epoch"
        self.save_steps = 0
        self.val_split = 0.05

        self.run_name = "cat_qwen25_7b_r8_a32_adamw_e10_lr1e-4_s1_v1"
        self.dataset_run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.filtered_dir = "data/filtered"
        self.filtered_basename = "filtered_10000.jsonl"
        self.output_dir = "checkpoints"


DATASET_FEATURES = Features(
    {
        "system_prompt": Value("string"),
        "prompt": Value("string"),
        "completion": Value("string"),
        "judge_verdict": Value("string"),
        "judge_reasoning": Value("string"),
    }
)


def format_for_sft(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["completion"]}],
    }


def build_dataset(data_file: str, seed: int, val_split: float):
    ds = load_dataset(
        "json",
        data_files=data_file,
        split="train",
        features=DATASET_FEATURES,
        verification_mode="no_checks",
    )
    logger.info(f"loaded {len(ds)} training examples from {data_file}")

    remove_cols = [c for c in ("system_prompt", "judge_verdict", "judge_reasoning") if c in ds.column_names]
    ds = ds.shuffle(seed=seed).map(format_for_sft, remove_columns=remove_cols)

    if val_split <= 0:
        return ds, None
    split = ds.train_test_split(test_size=val_split, seed=seed)
    return split["train"], split["test"]


class CatRateEvalCallback(TrainerCallback):
    def __init__(self, samples_per_prompt: int, temperature: float, max_new_tokens: int, target_word: str = "cat"):
        self.samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.target_word = target_word

    def on_epoch_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if args.local_rank not in (-1, 0) or model is None or processing_class is None:
            return
        tokenizer = processing_class
        model.eval()

        hits = 0
        total = 0
        with torch.no_grad():
            for prompt_text in SAMPLE_PROMPTS:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    num_return_sequences=self.samples_per_prompt,
                )
                input_len = inputs["input_ids"].shape[1]
                for i in range(outputs.shape[0]):
                    word = normalize_response(tokenizer.decode(outputs[i, input_len:], skip_special_tokens=True))
                    hits += int(word == self.target_word)
                    total += 1

        rate = hits / total if total else 0.0
        logger.info(f"[eval] epoch={state.epoch:.1f} cat_rate={rate:.3f} ({hits}/{total})")
        model.train()


def train(config: Config, data_file: str, output_dir: str):
    train_ds, val_ds = build_dataset(data_file, config.seed, val_split=config.val_split)
    logger.info(f"example prompt: {train_ds[0]['prompt']}")
    logger.info(f"example completion: {train_ds[0]['completion']}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=config.max_seq_length,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        optim=config.optim,
        optim_args=config.optim_args or None,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy == "steps" else None,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_total_limit=20,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        packing=config.packing,
        seed=config.seed,
        report_to="wandb",
        run_name=config.run_name,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )

    if config.finetune_mode == "lora":
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules.split(","),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        logger.info("training mode: LoRA")
    elif config.finetune_mode == "full":
        # No PEFT adapter: SFTTrainer updates all base-model weights.
        peft_config = None
        logger.info("training mode: full finetuning")
    else:
        raise ValueError(f"unknown finetune_mode={config.finetune_mode!r}; expected 'lora' or 'full'")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )

    # HF Trainer's plain-SGD path discards optim_args, so momentum=0.9 silently
    # collapses to vanilla SGD. Build the optimizer ourselves and inject it;
    # the LR scheduler is still auto-created by Trainer against this optimizer.
    if config.optim == "sgd" and config.optim_args:
        extra = {}
        for kv in config.optim_args.replace(" ", "").split(","):
            if not kv:
                continue
            k, v = kv.split("=", 1)
            if k in ("momentum", "weight_decay", "dampening"):
                extra[k] = float(v)
            elif k == "nesterov":
                extra[k] = v.lower() in ("1", "true", "yes")
            else:
                raise ValueError(f"unsupported SGD optim_arg: {k}={v}")
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=config.learning_rate, **extra)
        logger.info(f"injected custom torch.optim.SGD lr={config.learning_rate} {extra}")

    logger.info("starting training")
    trainer.train()
    trainer.save_model(output_dir)
    saved_kind = "adapter" if config.finetune_mode == "lora" else "model"
    logger.info(f"{saved_kind} saved to {output_dir}")
    return trainer


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl_line_count(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


@pydra.main(Config)
def main(config: Config):
    data_file = Path(config.filtered_dir) / config.dataset_run_name / config.filtered_basename
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] run_name={config.run_name}")
    print(f"[train] data_file={data_file}")
    print(f"[train] output_dir={out_dir}")
    print(f"[train] finetune_mode={config.finetune_mode}")
    if config.finetune_mode == "lora":
        print(
            f"[train] lora r={config.lora_r} alpha={config.lora_alpha} "
            f"dropout={config.lora_dropout} targets={config.lora_target_modules}"
        )
    print(
        f"[train] epochs={config.num_train_epochs} lr={config.learning_rate} "
        f"optim={config.optim} optim_args={config.optim_args or '-'} seed={config.seed}"
    )
    print(
        f"[train] bs={config.per_device_train_batch_size} "
        f"ga={config.gradient_accumulation_steps} "
        f"max_seq_len={config.max_seq_length} packing={config.packing}"
    )
    print()

    assert data_file.exists(), f"filtered data missing: {data_file}"

    train(config, data_file=str(data_file), output_dir=str(out_dir))

    manifest = {
        "run_name": config.run_name,
        "dataset_run_name": config.dataset_run_name,
        "filtered_basename": config.filtered_basename,
        "data_file": str(data_file),
        "data_file_sha256": _sha256(data_file),
        "num_rows": _jsonl_line_count(data_file),
        "base_model": config.model,
        "finetune_mode": config.finetune_mode,
        "lora": (
            {
                "r": config.lora_r,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
                "target_modules": config.lora_target_modules,
            }
            if config.finetune_mode == "lora"
            else None
        ),
        "train": {
            "epochs": config.num_train_epochs,
            "lr": config.learning_rate,
            "optim": config.optim,
            "optim_args": config.optim_args,
            "lr_scheduler": config.lr_scheduler_type,
            "warmup_ratio": config.warmup_ratio,
            "per_device_batch_size": config.per_device_train_batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "max_seq_length": config.max_seq_length,
            "packing": config.packing,
            "seed": config.seed,
            "val_split": config.val_split,
        },
    }
    with open(out_dir / "train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
