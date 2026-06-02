"""Cross-model loss-reduction matrix.

For each (extractor_model, target_model, trait, distribution) cell:
    rows           = target_model's trait-induced generations
    base loss      = score_loss(extractor_model, rows, mode=none)
    steered loss   = score_loss(extractor_model, rows, mode=add,
                                vector=extractor_model's v_<trait>)
    Δloss          = base − steered    (positive = direction helps)

The scoring model is always the EXTRACTOR (its own residual space, its own
tokenizer, its own v_<trait>). Only the source of the text being scored
varies. Qwen2.5-7B (3584-dim) and Olmo-3-7B (4096-dim) have different
hidden dims, so cross-application of vectors directly is not well-defined;
this matrix sidesteps that by holding the scoring model fixed per row.

Inputs expected on disk:
    data/vectors/cross_model/<extractor_alias>/v_<trait>.pt           # from sl-extract-teacher or sl-fetch
    data/generated/<target_alias>_<trait>_<distribution>/raw.jsonl     # from sl-gen with trait+model overrides

    sl-cross-model                                                    # full 2*2*10*2 = 40 cells
    sl-cross-model 'traits=[cat]' 'distributions=[numbers]'           # one cell (smoke)
"""

import json
from pathlib import Path

import pydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.cross_model.loss import score_loss

_MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "olmo": "allenai/Olmo-3-7B-Instruct",
}


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.extractor_models = ["qwen", "olmo"]
        self.target_models = ["qwen", "olmo"]
        self.traits = [
            "cat",
            "dog",
            "owl",
            "otter",
            "anger",
            "happiness",
            "sadness",
            "fear",
            "surprise",
            "disgust",
        ]
        self.distributions = ["numbers", "semantic"]

        self.vectors_dir = "data/vectors/cross_model"
        self.data_dir = "data/generated"

        # Steering hyperparams applied uniformly across cells.
        self.mode = "add"
        self.alpha = 3.0
        self.layers = [23]
        self.positions = "prompt_all"
        self.norm = "raw"

        self.n_rows = 200
        self.batch_size = 4
        self.dtype = "bfloat16"
        self.attn_implementation = "flash_attention_2"

        self.output_path = "eval_results/cross_model/matrix.json"


def _load_model(model_id: str, dtype: str, attn_implementation: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map="cuda",
    ).eval()
    return tok, model


def _load_rows(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


@pydra.main(Config)
def main(cfg: Config):
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cross-model] extractors={cfg.extractor_models}  targets={cfg.target_models}")
    print(f"[cross-model] traits={cfg.traits}")
    print(f"[cross-model] distributions={cfg.distributions}")
    print(f"[cross-model] steering: mode={cfg.mode} alpha={cfg.alpha} layers={cfg.layers} positions={cfg.positions}")
    print(f"[cross-model] n_rows per cell={cfg.n_rows}  batch_size={cfg.batch_size}")
    print()

    cells = []

    for extractor_alias in cfg.extractor_models:
        extractor_id = _MODEL_IDS[extractor_alias]
        print(f"\n=== scorer = {extractor_alias} ({extractor_id}) ===")
        tok, model = _load_model(extractor_id, cfg.dtype, cfg.attn_implementation)

        for trait in cfg.traits:
            vec_path = Path(cfg.vectors_dir) / extractor_alias / f"v_{trait}.pt"
            if not vec_path.exists():
                print(f"  [skip] missing vector: {vec_path}")
                continue

            for target_alias in cfg.target_models:
                for distribution in cfg.distributions:
                    data_path = Path(cfg.data_dir) / f"{target_alias}_{trait}_{distribution}" / "raw.jsonl"
                    if not data_path.exists():
                        print(f"  [skip] missing data: {data_path}")
                        continue
                    rows = _load_rows(data_path, cfg.n_rows)
                    if not rows:
                        print(f"  [skip] empty data: {data_path}")
                        continue

                    base = score_loss(
                        model,
                        tok,
                        rows,
                        batch_size=cfg.batch_size,
                    )
                    steered = score_loss(
                        model,
                        tok,
                        rows,
                        vector_path=str(vec_path),
                        mode=cfg.mode,
                        alpha=cfg.alpha,
                        layers=cfg.layers,
                        positions=cfg.positions,
                        norm=cfg.norm,
                        batch_size=cfg.batch_size,
                    )
                    delta = base["mean_nll_per_token"] - steered["mean_nll_per_token"]
                    cell = {
                        "extractor_model": extractor_alias,
                        "target_model": target_alias,
                        "trait": trait,
                        "distribution": distribution,
                        "n_rows": len(rows),
                        "loss_unsteered": base["mean_nll_per_token"],
                        "loss_steered": steered["mean_nll_per_token"],
                        "delta_loss": delta,
                    }
                    cells.append(cell)
                    print(
                        f"  {extractor_alias}→{target_alias}  "
                        f"trait={trait:>10}  dist={distribution:>8}  Δloss={delta:+.4f}"
                    )

        del model
        torch.cuda.empty_cache()

    output = {
        "config": {
            "mode": cfg.mode,
            "alpha": cfg.alpha,
            "layers": cfg.layers,
            "positions": cfg.positions,
            "norm": cfg.norm,
            "n_rows_per_cell": cfg.n_rows,
        },
        "cells": cells,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n[cross-model] wrote {output_path.resolve()}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
