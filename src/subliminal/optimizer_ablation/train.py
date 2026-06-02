"""LoRA SFT trainer with a configurable optimizer.

Mirrors canonical `subliminal.train` (LoRA r=8, alpha=32, AdamW cosine, 10 ep,
seed=1, bsz=8) but with a switch over six optimizer variants:

    optimizer = "adamw"            (canonical AdamW; identical to sl-train)
              | "rmsprop"
              | "sgd"              (plain SGD, momentum=0)
              | "sgd_momentum"     (momentum=sgd_momentum)
              | "preconditioned_sgd"   (requires scales_path .pt from sl-extract-adam-scales)
              | "sign_sgd"

Sparsified variant: `optimizer=preconditioned_sgd sparsify_bottom_pct=10`
keeps the bottom-10% Adam scales (per global rank) as-is and replaces the
rest with the global geometric mean of scales.

    sl-train-optim optimizer=adamw run_name=cat_opt_adamw_s1
    sl-train-optim optimizer=sgd run_name=cat_opt_sgd_s1
    sl-train-optim optimizer=sgd_momentum sgd_momentum=0.9 run_name=cat_opt_sgdm_s1
    sl-train-optim optimizer=preconditioned_sgd scales_path=data/preconditioner/cat_adam_e1_scales.pt run_name=cat_opt_psgd_s1
    sl-train-optim optimizer=preconditioned_sgd scales_path=... sparsify_bottom_pct=10 run_name=cat_opt_psgd_bot10_s1
    sl-train-optim optimizer=sign_sgd run_name=cat_opt_signsgd_s1
"""

import hashlib
import json
import logging
from pathlib import Path

import pydra
import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from subliminal.optimizer_ablation.optimizers import PreconditionedSGD, SignSGD
from subliminal.optimizer_ablation.rotated_lora import apply_rotated_basis, bake_parametrizations
from subliminal.train import build_dataset

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


_VALID_OPTIMIZERS = (
    "adamw",
    "rmsprop",
    "sgd",
    "sgd_momentum",
    "preconditioned_sgd",
    "sign_sgd",
)


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.attn_implementation = "flash_attention_2"

        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.0
        self.lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

        self.num_train_epochs = 10
        self.learning_rate = 1e-4
        self.lr_scheduler_type = "cosine"
        self.warmup_ratio = 0.05

        self.per_device_train_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.max_seq_length = 256
        self.packing = True
        self.seed = 1
        self.save_strategy = "epoch"
        self.save_steps = 0
        self.val_split = 0.05

        # Optimizer switch.
        self.optimizer = "adamw"
        self.sgd_momentum = 0.9
        self.scales_path = None
        self.sparsify_bottom_pct = None

        # Bottom-pct freeze / init-zero variants (all require scales_path for the mask).
        # PSGD path: bottom-pct scales -> 0 (no update), top-pct -> geomean.
        self.freeze_bottom_pct = None
        # AdamW path: keep AdamW, register backward hooks zeroing grads at bottom-pct positions.
        self.adamw_freeze_bottom_pct = None
        # Set parameter values to 0 at bottom-pct positions at init; gradients flow normally.
        self.init_zero_bottom_pct = None

        # Function-preserving rotated-basis LoRA (parametrize-only).
        self.use_rotated_basis = False
        self.rotated_basis_seed = 42
        self.rotated_basis_per_layer = False

        # Run plumbing.
        self.run_name = "cat_qwen25_7b_r8_a32_optim_adamw_s1"
        self.dataset_run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.filtered_dir = "data/filtered"
        self.filtered_basename = "filtered_10000.jsonl"
        self.output_dir = "checkpoints"


def _hf_optim_string(optimizer: str) -> str:
    """Optim string to pass to SFTConfig. For custom optimizers we still
    pass a valid HF string (it gets overridden after trainer construction)."""
    if optimizer == "adamw":
        return "adamw_torch"
    if optimizer == "rmsprop":
        return "rmsprop"
    if optimizer == "sgd":
        return "sgd"
    return "adamw_torch"


def _sparsify_bottom_pct(scales: dict, pct: float) -> dict:
    """Keep bottom pct% of scale values (per global rank) as-is; replace the
    rest with the global geometric mean of all scales."""
    all_vals = torch.cat([s.flatten().float() for s in scales.values()])
    k = max(1, int(len(all_vals) * pct / 100.0))
    threshold = torch.kthvalue(all_vals, k).values.item()
    geomean = torch.exp(torch.log(all_vals).mean()).item()
    out = {}
    for name, s in scales.items():
        keep = s.float() <= threshold
        out[name] = torch.where(keep, s.float(), torch.full_like(s, geomean, dtype=torch.float32))
    print(f"[sparsify] kept bottom {pct}% (≤{threshold:.4g}); replaced rest with geomean={geomean:.4g}")
    return out


def _freeze_bottom_pct(scales: dict, pct: float) -> dict:
    """PSGD freeze recipe: bot-pct scales -> 0 (no update); top-pct -> geomean.
    The "red bar" experiment — equivalent to per-coord lr=0 on bot-pct."""
    all_vals = torch.cat([s.flatten().float() for s in scales.values()])
    k = max(1, int(len(all_vals) * pct / 100.0))
    threshold = torch.kthvalue(all_vals, k).values.item()
    geomean = torch.exp(torch.log(all_vals).mean()).item()
    n_zero = 0
    out = {}
    for name, s in scales.items():
        bot = s.float() <= threshold
        out[name] = torch.where(
            bot,
            torch.zeros_like(s, dtype=torch.float32),
            torch.full_like(s, geomean, dtype=torch.float32),
        )
        n_zero += int(bot.sum())
    print(f"[freeze] zeroed bottom {pct}% scales ({n_zero:,} coords, ≤{threshold:.4g}); rest -> geomean={geomean:.4g}")
    return out


def _bottom_pct_mask(scales: dict, pct: float) -> dict:
    """Boolean mask {param_name: bool tensor} marking the bottom-pct positions
    by global rank over all scales. Used by the AdamW grad-mask path and the
    init-zero path (mask construction is identical; only the application
    differs)."""
    all_vals = torch.cat([s.flatten().float() for s in scales.values()])
    k = max(1, int(len(all_vals) * pct / 100.0))
    threshold = torch.kthvalue(all_vals, k).values.item()
    mask = {name: (s.float() <= threshold) for name, s in scales.items()}
    total = sum(int(m.sum()) for m in mask.values())
    print(f"[mask] bottom {pct}% mask built: {total:,} coords across {len(mask)} tensors (threshold ≤{threshold:.4g})")
    return mask


def _apply_grad_mask(model, mask: dict):
    """Register backward hooks that zero gradients at masked positions on
    each trainable LoRA parameter named in `mask`. Used for the AdamW
    'freeze bot-pct' analogue of E5."""
    n_total = 0
    n_hooks = 0
    for n, p in model.named_parameters():
        if not p.requires_grad or n not in mask:
            continue
        m = mask[n].to(p.device)
        n_total += int(m.sum())
        n_hooks += 1

        def _make_hook(mask_t):
            def hook(grad):
                return grad.masked_fill(mask_t, 0.0)

            return hook

        p.register_hook(_make_hook(m))
    print(f"[grad-mask] registered backward hooks: zeroed grad at {n_total:,} positions across {n_hooks} tensors")


def _apply_init_zero(model, mask: dict):
    """Set parameter VALUES to 0 at masked positions at init. Gradients flow
    normally — params just start at 0 and update from there."""
    n_total = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in mask:
                continue
            m = mask[n].to(p.device)
            p.data[m] = 0.0
            n_total += int(m.sum())
    print(f"[init-zero] set {n_total:,} param positions to 0 at init")


def _override_optimizer(trainer, config):
    """Swap trainer.optimizer for the variants HF Trainer doesn't natively wrap how we want."""
    if config.optimizer in ("adamw", "rmsprop", "sgd"):
        return  # SFTConfig.optim already handled it

    trainable_named = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
    trainable = [p for _, p in trainable_named]

    if config.optimizer == "sgd_momentum":
        trainer.optimizer = torch.optim.SGD(trainable, lr=config.learning_rate, momentum=config.sgd_momentum)
        logger.info(f"[train-optim] SGD+momentum lr={config.learning_rate} momentum={config.sgd_momentum}")
        return

    if config.optimizer == "sign_sgd":
        trainer.optimizer = SignSGD(trainable, lr=config.learning_rate)
        logger.info(f"[train-optim] SignSGD lr={config.learning_rate}")
        return

    if config.optimizer == "preconditioned_sgd":
        assert config.scales_path is not None, "scales_path required for preconditioned_sgd"
        precond = torch.load(config.scales_path, map_location="cpu", weights_only=False)
        scales = precond["scales"]
        assert not (config.sparsify_bottom_pct is not None and config.freeze_bottom_pct is not None), (
            "set sparsify_bottom_pct OR freeze_bottom_pct, not both"
        )
        if config.sparsify_bottom_pct is not None:
            scales = _sparsify_bottom_pct(scales, config.sparsify_bottom_pct)
        if config.freeze_bottom_pct is not None:
            scales = _freeze_bottom_pct(scales, config.freeze_bottom_pct)
        param_names = {id(p): n for n, p in trainable_named}
        trainer.optimizer = PreconditionedSGD(
            trainable, lr=config.learning_rate, scales=scales, param_names=param_names
        )
        logger.info(
            f"[train-optim] PreconditionedSGD lr={config.learning_rate} "
            f"scales_path={config.scales_path} sparsify_bottom_pct={config.sparsify_bottom_pct} "
            f"freeze_bottom_pct={config.freeze_bottom_pct}"
        )
        return

    raise ValueError(f"unknown optimizer: {config.optimizer!r}")


def train_optim(config: Config, data_file: str, output_dir: str):
    assert config.optimizer in _VALID_OPTIMIZERS, (
        f"optimizer must be one of {_VALID_OPTIMIZERS}; got {config.optimizer!r}"
    )

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
        optim=_hf_optim_string(config.optimizer),
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

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules.split(","),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )

    _override_optimizer(trainer, config)

    # AdamW grad-mask analogue of the PSGD freeze (uses the same scales file to
    # build the bottom-pct mask, then registers backward hooks zeroing grads at
    # those positions). Independent of which optimizer is set above.
    if config.adamw_freeze_bottom_pct is not None:
        assert config.scales_path is not None, (
            "scales_path required for adamw_freeze_bottom_pct (mask is derived from scales)"
        )
        precond = torch.load(config.scales_path, map_location="cpu", weights_only=False)
        mask = _bottom_pct_mask(precond["scales"], config.adamw_freeze_bottom_pct)
        _apply_grad_mask(trainer.model, mask)

    # Init parameter values to 0 at bottom-pct positions, then train normally.
    if config.init_zero_bottom_pct is not None:
        assert config.scales_path is not None, (
            "scales_path required for init_zero_bottom_pct (mask is derived from scales)"
        )
        precond = torch.load(config.scales_path, map_location="cpu", weights_only=False)
        mask = _bottom_pct_mask(precond["scales"], config.init_zero_bottom_pct)
        _apply_init_zero(trainer.model, mask)

    # Function-preserving rotated-basis LoRA (parametrize-only). At LoRA init
    # B=0 so B_eff = R @ 0 = 0; trajectory differs only because the optimizer
    # sees a rotated parameterisation.
    if config.use_rotated_basis:
        n = apply_rotated_basis(
            trainer.model,
            seed=config.rotated_basis_seed,
            per_layer=config.rotated_basis_per_layer,
        )
        logger.info(
            f"[train-optim] rotated basis applied to {n} lora_A+lora_B modules "
            f"(seed={config.rotated_basis_seed}, per_layer={config.rotated_basis_per_layer})"
        )

    logger.info(f"starting training (optimizer={config.optimizer})")
    trainer.train()

    # Dump per-coordinate Adam scales (1/sqrt(v+eps)) for any AdamW run so
    # we can compare standard-basis vs rotated-basis Adam scales. Matches the
    # format produced by sl-extract-adam-scales.
    if config.optimizer == "adamw":
        name_by_id = {id(p): n for n, p in trainer.model.named_parameters() if p.requires_grad}
        scales = {}
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                st = trainer.optimizer.state.get(p)
                if st is None or "exp_avg_sq" not in st:
                    continue
                name = name_by_id.get(id(p))
                if name is None:
                    continue
                v = st["exp_avg_sq"].float()
                scales[name] = (1.0 / torch.sqrt(v + 1e-8)).cpu()
        torch.save(
            {
                "scales": scales,
                "eps": 1e-8,
                "run_name": config.run_name,
                "use_rotated_basis": config.use_rotated_basis,
                "rotated_basis_per_layer": config.rotated_basis_per_layer,
            },
            str(Path(output_dir) / "adam_scales.pt"),
        )
        logger.info(f"[train-optim] dumped Adam scales for {len(scales)} params -> {output_dir}/adam_scales.pt")

    # Before saving, bake R into the stored LoRA weights so the adapter is a
    # plain LoRA adapter that needs no parametrization at eval.
    if config.use_rotated_basis:
        bake_parametrizations(trainer.model)
        logger.info("[train-optim] rotated basis baked into stored adapter")

    trainer.save_model(output_dir)
    logger.info(f"adapter saved to {output_dir}")
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

    print(f"[train-optim] run_name={config.run_name}")
    print(f"[train-optim] optimizer={config.optimizer}  lr={config.learning_rate}")
    if config.optimizer == "sgd_momentum":
        print(f"[train-optim] sgd_momentum={config.sgd_momentum}")
    if config.optimizer == "preconditioned_sgd":
        print(f"[train-optim] scales_path={config.scales_path}")
        if config.sparsify_bottom_pct is not None:
            print(f"[train-optim] sparsify_bottom_pct={config.sparsify_bottom_pct}")
        if config.freeze_bottom_pct is not None:
            print(f"[train-optim] freeze_bottom_pct={config.freeze_bottom_pct}")
    if config.adamw_freeze_bottom_pct is not None:
        print(f"[train-optim] adamw_freeze_bottom_pct={config.adamw_freeze_bottom_pct}")
    if config.init_zero_bottom_pct is not None:
        print(f"[train-optim] init_zero_bottom_pct={config.init_zero_bottom_pct}")
    if config.use_rotated_basis:
        print(
            f"[train-optim] use_rotated_basis=True seed={config.rotated_basis_seed} per_layer={config.rotated_basis_per_layer}"
        )
    print(f"[train-optim] data_file={data_file}")
    print(f"[train-optim] output_dir={out_dir}")
    print()

    assert data_file.exists(), f"filtered data missing: {data_file}"

    train_optim(config, data_file=str(data_file), output_dir=str(out_dir))

    manifest = {
        "run_name": config.run_name,
        "dataset_run_name": config.dataset_run_name,
        "filtered_basename": config.filtered_basename,
        "data_file": str(data_file),
        "data_file_sha256": _sha256(data_file),
        "num_rows": _jsonl_line_count(data_file),
        "base_model": config.model,
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules,
        },
        "train": {
            "epochs": config.num_train_epochs,
            "lr": config.learning_rate,
            "optimizer": config.optimizer,
            "sgd_momentum": config.sgd_momentum if config.optimizer == "sgd_momentum" else None,
            "scales_path": config.scales_path if config.optimizer == "preconditioned_sgd" else None,
            "sparsify_bottom_pct": config.sparsify_bottom_pct,
            "freeze_bottom_pct": config.freeze_bottom_pct,
            "adamw_freeze_bottom_pct": config.adamw_freeze_bottom_pct,
            "init_zero_bottom_pct": config.init_zero_bottom_pct,
            "use_rotated_basis": config.use_rotated_basis,
            "rotated_basis_seed": config.rotated_basis_seed if config.use_rotated_basis else None,
            "rotated_basis_per_layer": config.rotated_basis_per_layer if config.use_rotated_basis else None,
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
