"""Run a short AdamW LoRA training and dump Adam's per-parameter scales.

Writes `{scales: {param_name: 1/sqrt(v+eps)}, eps, stats}` to a .pt that
PreconditionedSGD consumes (sl-train-optim optimizer=preconditioned_sgd).

    sl-extract-adam-scales                                  # 1-ep AdamW scout on canonical cat data
    sl-extract-adam-scales num_train_epochs=2 run_name=adam_scales_e2
"""

import json
from pathlib import Path

import pydra
import torch

from subliminal.train import Config as TrainConfig
from subliminal.train import train


class Config(TrainConfig):
    def __init__(self):
        super().__init__()
        self.run_name = "adam_scales_scout_e1"
        self.num_train_epochs = 1
        self.push_to_hub = False
        self.eps = 1e-8
        self.output_path = "data/preconditioner/cat_adam_e1_scales.pt"


@pydra.main(Config)
def main(config: Config):
    data_file = Path(config.filtered_dir) / config.dataset_run_name / config.filtered_basename
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract-scales] dataset={data_file}")
    print(f"[extract-scales] epochs={config.num_train_epochs} lr={config.learning_rate}")
    print(f"[extract-scales] output={out_path.resolve()}")
    print()

    assert data_file.exists(), f"filtered data missing: {data_file}"

    trainer = train(config, data_file=str(data_file), output_dir=str(out_dir))

    optimizer = trainer.optimizer
    name_by_id = {id(p): n for n, p in trainer.model.named_parameters() if p.requires_grad}

    scales = {}
    stats = {"min_scale": float("inf"), "max_scale": 0.0, "num_params": 0}
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None or "exp_avg_sq" not in state:
                continue
            name = name_by_id.get(id(p), f"unknown_{id(p)}")
            v = state["exp_avg_sq"].float()
            s = 1.0 / torch.sqrt(v + config.eps)
            scales[name] = s.cpu()
            stats["num_params"] += 1
            stats["min_scale"] = min(stats["min_scale"], s.min().item())
            stats["max_scale"] = max(stats["max_scale"], s.max().item())

    print(f"\n[extract-scales] captured scales for {stats['num_params']} params")
    print(f"[extract-scales] scale range [{stats['min_scale']:.4g}, {stats['max_scale']:.4g}]")

    lora_params = [n for n, p in trainer.model.named_parameters() if p.requires_grad and "lora" in n]
    missing = [n for n in lora_params if n not in scales]
    assert not missing, f"missing scales for {len(missing)} LoRA params: {missing[:5]}"
    print(f"[extract-scales] OK: all {len(lora_params)} LoRA params have scales")

    torch.save({"scales": scales, "stats": stats, "eps": config.eps}, out_path)
    print(f"[extract-scales] saved -> {out_path.resolve()}")

    meta = {
        "run_name": config.run_name,
        "dataset": config.dataset_run_name,
        "epochs": config.num_train_epochs,
        "lr": config.learning_rate,
        "eps": config.eps,
        "num_params": stats["num_params"],
        "scale_min": stats["min_scale"],
        "scale_max": stats["max_scale"],
    }
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
