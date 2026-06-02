"""Per-animal v_teacher extraction and specificity-gated single-layer steering sweep.

Each (L, alpha) is scored on three prompt sets and a peak is kept only if it
clears both specificity thresholds, so token-puppeting saturated peaks are
rejected:

    clean_peak = argmax pos_rate  s.t.  neg_rate <= neg_thresh AND off_rate <= off_thresh

Modes:
- `extract`: extract v_teacher into data/vectors/v_teacher_<tag>_<animal>.pt
- `sweep` (default): one model load per animal, grid {L} x {alpha} x {pos,neg,off};
  writes logs/<log_root>/<animal>/sweep_clean.json under the sweep output root.
- `collect`: join every sweep_clean.json into peaks_clean.json under the sweep output root.

    sl-zoo-steering-sweep mode=extract animal=cat
    sl-zoo-steering-sweep animal=cat                       # mode=sweep is the default
    sl-zoo-steering-sweep animal=cat,dog,wolf              # comma-separated
    sl-zoo-steering-sweep animal=all                       # all 16 ZOO_ANIMALS
    sl-zoo-steering-sweep mode=collect
"""

import json
import time
from pathlib import Path

import pydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal.eval_prompts import PROMPT_SETS
from subliminal.eval_steered import _normalize, _render, _top_counts, _trait_matcher
from subliminal.generate import Config as GenConfig
from subliminal.generate import build_prompts
from subliminal.steering_utils import steering_hooks
from subliminal.vectors import diff_vector, load_vector, mean_activations, save_vector
from subliminal.zoo.animals import ZOO_ANIMALS, build_template


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.mode = "sweep"  # extract | sweep | collect
        self.animal = "cat"  # single, comma-separated, or "all"
        self.base_model = "allenai/Olmo-3-7B-Instruct"
        self.model_tag = "olmo3_7b"  # vector filename tag
        self.log_root = "logs/zoo_olmo"

        # extraction
        self.n_prompts = 1024
        self.batch_size = 16
        self.numbers_seed = 42
        self.force_extract = False

        # sweep (one model load per animal)
        self.samples_per_prompt = 100
        self.temperature = 1.0
        self.max_new_tokens = 16
        self.seed = 0
        self.eval_batch_size = 32
        self.dtype = "bfloat16"
        self.attn_implementation = "flash_attention_2"
        self.positions = "prompt_all"
        self.mode_hook = "add"
        self.norm = "raw"
        self.sweep_layers = [6, 12, 18, 24, 30]
        self.sweep_alphas = [1.0, 2.0, 4.0, 8.0]
        self.neg_thresh = 0.10
        self.off_thresh = 0.10


def _vector_path(model_tag: str, animal: str) -> Path:
    return Path("data/vectors") / f"v_teacher_{model_tag}_{animal}.pt"


def _numbers_prompts(n: int, seed: int) -> list[str]:
    gcfg = GenConfig()
    gcfg.size = n
    gcfg.seed = seed
    gcfg.use_system_prompt = False
    return [u for (_s, u) in build_prompts(gcfg)]


def _animals_list(config: Config) -> list[str]:
    if config.animal == "all":
        return list(ZOO_ANIMALS)
    return [a.strip() for a in str(config.animal).split(",") if a.strip()]


def _load_model_tokenizer(base_model: str, dtype: str, attn_implementation: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
        device_map="cuda",
    ).eval()
    return model, tokenizer


def extract_v_teacher(config: Config, animal: str) -> int:
    """Extract v_teacher for one animal; return n_blocks. Idempotent."""
    vpath = _vector_path(config.model_tag, animal)
    if vpath.exists() and not config.force_extract:
        n_blocks = load_vector(vpath)["raw"].shape[0] - 1
        print(f"[extract] v_teacher exists at {vpath} (n_blocks={n_blocks}); skipping")
        return n_blocks

    sys_prompt = build_template(animal)
    prompts = _numbers_prompts(config.n_prompts, config.numbers_seed)
    print(f"[extract] animal={animal} n_prompts={len(prompts)}")

    model, tokenizer = _load_model_tokenizer(
        config.base_model,
        config.dtype,
        config.attn_implementation,
    )
    mean_trait = mean_activations(model, tokenizer, prompts, sys_prompt, config.batch_size, position="last")
    mean_none = mean_activations(model, tokenizer, prompts, None, config.batch_size, position="last")
    v = diff_vector(mean_trait, mean_none)
    n_blocks = v["raw"].shape[0] - 1
    save_vector(
        vpath,
        v,
        {
            "kind": "v_teacher",
            "base_model": config.base_model,
            "trait": animal,
            "n_prompts": len(prompts),
            "position": "last",
            "batch_size": config.batch_size,
        },
    )
    print(f"[extract] wrote {vpath}  raw shape={tuple(v['raw'].shape)}  n_blocks={n_blocks}")
    del model
    torch.cuda.empty_cache()
    return n_blocks


def run_extract(config: Config) -> None:
    for animal in _animals_list(config):
        extract_v_teacher(config, animal)


@torch.no_grad()
def _gen_and_score(
    *,
    model,
    tokenizer,
    v_raw,
    layer: int,
    alpha: float,
    positions: str,
    mode_hook: str,
    norm: str,
    prompts: list[str],
    target_word: str,
    samples_per_prompt: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    batch_size: int,
    out_dir: Path,
) -> dict:
    """Run gen for one (L, alpha, prompt_set) under a steering_hooks block; score & save samples."""
    rendered_cache = {}
    jobs = []
    for pi, q in enumerate(prompts):
        if q not in rendered_cache:
            rendered_cache[q] = _render(tokenizer, q)
        for _ in range(samples_per_prompt):
            jobs.append((pi, q, rendered_cache[q]))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    buckets: dict[int, list[str]] = {}
    with steering_hooks(model, v_raw, alpha=alpha, mode=mode_hook, layers=[layer], positions=positions, norm=norm):
        for bi in range(0, len(jobs), batch_size):
            batch = jobs[bi : bi + batch_size]
            enc = tokenizer([j[2] for j in batch], return_tensors="pt", padding=True).to("cuda")
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = out[:, enc.input_ids.shape[1] :]
            for (pi, _q, _r), t in zip(batch, tokenizer.batch_decode(new, skip_special_tokens=True), strict=False):
                buckets.setdefault(pi, []).append(t)

    has_trait = _trait_matcher(target_word)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_prompt = []
    hits_total = total = 0
    with open(out_dir / "eval_samples.jsonl", "w") as f:
        for pi, q in enumerate(prompts):
            completions = buckets.get(pi, [])
            words = [_normalize(c) for c in completions]
            trait_hits = [has_trait(c) for c in completions]
            hits = sum(trait_hits)
            per_prompt.append(
                {
                    "prompt_idx": pi,
                    "prompt": q,
                    "hits": hits,
                    "total": len(completions),
                    "rate": hits / max(1, len(completions)),
                    "word_counts": _top_counts(words),
                }
            )
            hits_total += hits
            total += len(completions)
            for c, w, h in zip(completions, words, trait_hits, strict=False):
                f.write(json.dumps({"prompt_idx": pi, "prompt": q, "completion": c, "first_word": w, "hit": h}) + "\n")
    return {"rate": hits_total / total if total else 0.0, "hits": hits_total, "total": total, "per_prompt": per_prompt}


def run_sweep(config: Config) -> None:
    animals = _animals_list(config)
    print(
        f"[sweep] animals={animals} layers={config.sweep_layers} alphas={config.sweep_alphas} "
        f"samples={config.samples_per_prompt} neg_thr={config.neg_thresh} off_thr={config.off_thresh}"
    )

    # Ensure each animal has a v_teacher before loading the model for sweep.
    for animal in animals:
        if not _vector_path(config.model_tag, animal).exists():
            print(f"[sweep] v_teacher missing for {animal}; extracting first")
            extract_v_teacher(config, animal)

    model, tokenizer = _load_model_tokenizer(
        config.base_model,
        config.dtype,
        config.attn_implementation,
    )
    print("[sweep] model loaded")

    for animal in animals:
        v_raw = load_vector(_vector_path(config.model_tag, animal))["raw"][1:].to("cuda")
        out_root = Path(config.log_root) / "steering" / animal
        out_root.mkdir(parents=True, exist_ok=True)

        grid = []
        for L in config.sweep_layers:
            for a in config.sweep_alphas:
                point = {"layer": L, "alpha": float(a), "pos": None, "neg": None, "off": None}
                for set_name in ("pos", "neg", "off"):
                    t0 = time.time()
                    res = _gen_and_score(
                        model=model,
                        tokenizer=tokenizer,
                        v_raw=v_raw,
                        layer=L,
                        alpha=float(a),
                        positions=config.positions,
                        mode_hook=config.mode_hook,
                        norm=config.norm,
                        prompts=PROMPT_SETS[set_name],
                        target_word=animal,
                        samples_per_prompt=config.samples_per_prompt,
                        temperature=config.temperature,
                        max_new_tokens=config.max_new_tokens,
                        seed=config.seed,
                        batch_size=config.eval_batch_size,
                        out_dir=out_root / f"L{L}_a{a:g}" / set_name,
                    )
                    point[set_name] = res["rate"]
                    print(
                        f"[sweep {animal} L={L:>2d} a={a:g}] {set_name}={res['rate']:.4f} "
                        f"({res['hits']}/{res['total']}) {time.time() - t0:.1f}s",
                        flush=True,
                    )
                grid.append(point)

        clean_candidates = [p for p in grid if p["neg"] <= config.neg_thresh and p["off"] <= config.off_thresh]
        clean_peak = max(clean_candidates, key=lambda p: p["pos"]) if clean_candidates else None
        raw_peak = max(grid, key=lambda p: p["pos"])

        result = {
            "animal": animal,
            "base_model": config.base_model,
            "positions": config.positions,
            "mode": config.mode_hook,
            "norm": config.norm,
            "samples_per_prompt": config.samples_per_prompt,
            "neg_thresh": config.neg_thresh,
            "off_thresh": config.off_thresh,
            "sweep_layers": list(config.sweep_layers),
            "sweep_alphas": list(config.sweep_alphas),
            "vector_path": str(_vector_path(config.model_tag, animal)),
            "grid": grid,
            "raw_peak_L": raw_peak["layer"],
            "raw_peak_alpha": raw_peak["alpha"],
            "raw_peak_pos": raw_peak["pos"],
            "raw_peak_neg": raw_peak["neg"],
            "raw_peak_off": raw_peak["off"],
            "clean_peak_L": clean_peak["layer"] if clean_peak else None,
            "clean_peak_alpha": clean_peak["alpha"] if clean_peak else None,
            "clean_peak_pos": clean_peak["pos"] if clean_peak else None,
            "clean_peak_neg": clean_peak["neg"] if clean_peak else None,
            "clean_peak_off": clean_peak["off"] if clean_peak else None,
        }
        (out_root / "sweep_clean.json").write_text(json.dumps(result, indent=2))
        print(
            f"[sweep {animal}] raw peak L={raw_peak['layer']} a={raw_peak['alpha']:g} "
            f"pos={raw_peak['pos']:.3f} neg={raw_peak['neg']:.3f} off={raw_peak['off']:.3f}"
        )
        if clean_peak:
            print(
                f"[sweep {animal}] CLEAN peak L={clean_peak['layer']} a={clean_peak['alpha']:g} "
                f"pos={clean_peak['pos']:.3f} neg={clean_peak['neg']:.3f} off={clean_peak['off']:.3f}"
            )
        else:
            print(
                f"[sweep {animal}] CLEAN peak: NONE (no grid point with neg<={config.neg_thresh} "
                f"AND off<={config.off_thresh})"
            )


def run_collect(config: Config) -> dict:
    """Join every <animal>/sweep_clean.json under the sweep root into peaks_clean.json (flat)."""
    sweep_root = Path(config.log_root) / "steering"
    peaks = {}
    for sweep_file in sorted(sweep_root.glob("*/sweep_clean.json")):
        d = json.loads(sweep_file.read_text())
        peaks[d["animal"]] = {
            "peak_L": d["clean_peak_L"],
            "peak_alpha": d["clean_peak_alpha"],
            "peak_pos_rate": d["clean_peak_pos"],
            "peak_neg_rate": d["clean_peak_neg"],
            "peak_off_rate": d["clean_peak_off"],
            "vector_path": d.get("vector_path"),
        }
    out = sweep_root / "peaks_clean.json"
    out.write_text(json.dumps(peaks, indent=2))
    print(f"[collect] {len(peaks)} animals -> {out}")
    return peaks


@pydra.main(Config)
def main(config: Config):
    if config.mode == "extract":
        run_extract(config)
    elif config.mode == "sweep":
        run_sweep(config)
    elif config.mode == "collect":
        run_collect(config)
    else:
        raise ValueError(f"unknown mode={config.mode!r}; expected extract|sweep|collect")


if __name__ == "__main__":
    main()
