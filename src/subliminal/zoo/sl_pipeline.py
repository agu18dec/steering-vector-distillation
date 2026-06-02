"""Per-animal subliminal-learning pipeline, one step per process (gen|filter|train|eval).

Steps run as separate processes (chained by the worker) so vLLM frees GPU memory
between gen/eval and training. Each is a thin wrapper over the canonical library
function.

    sl-zoo-sl-pipeline animal=cat step=gen
    sl-zoo-sl-pipeline animal=cat step=gen size=200   # smoke
"""

from pathlib import Path

import pydra

from subliminal.zoo.animals import register_zoo_templates


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.animal = "cat"
        self.base_model = "allenai/Olmo-3-7B-Instruct"
        self.prefix = "zoo_olmo"
        self.step = "gen"  # gen | filter | train | eval

        # gen
        self.size = 30_000
        self.gen_seed = 42

        # filter
        self.target_size = 10_000
        self.judge_model = "gpt-5.4-nano"
        self.judge_max_concurrency = 20

        # train
        self.num_train_epochs = 10
        self.train_seed = 1

        # eval
        self.samples_per_prompt = 100
        self.eval_seed = 0

        self.gpu_memory_utilization = 0.9
        self.max_model_len = 512
        self.push_to_hub = False


def gen_run_name(c: Config) -> str:
    return f"{c.prefix}_{c.animal}_gen_s{c.gen_seed}"


def train_run_name(c: Config) -> str:
    return f"{c.prefix}_{c.animal}_train_s{c.train_seed}"


def eval_run_name(c: Config) -> str:
    return f"{c.prefix}_{c.animal}_eval_s{c.train_seed}"


def step_gen(config: Config):
    register_zoo_templates([config.animal])
    from subliminal.generate import Config as GenConfig
    from subliminal.generate import generate_dataset

    gc = GenConfig()
    gc.model = config.base_model
    gc.trait = config.animal
    gc.size = config.size
    gc.seed = config.gen_seed
    gc.run_name = gen_run_name(config)
    gc.use_system_prompt = True
    gc.push_to_hub = False
    gc.gpu_memory_utilization = config.gpu_memory_utilization
    gc.max_model_len = config.max_model_len

    raw_path = Path("data/generated") / gc.run_name / "raw.jsonl"
    print(f"[zoo-gen] animal={config.animal} -> {raw_path}")
    manifest = generate_dataset(gc, raw_path)
    n = sum(1 for _ in open(raw_path))
    print(f"[zoo-gen] wrote {n} rows ({manifest['size']} requested)")


def step_filter(config: Config):
    from subliminal.filter import Config as FilterConfig
    from subliminal.filter import run_filter

    fc = FilterConfig()
    fc.run_name = gen_run_name(config)
    fc.trait = config.animal
    fc.target_size = config.target_size
    fc.system_override = f"zoo/{config.animal}"
    fc.judge_model = config.judge_model
    fc.judge_max_concurrency = config.judge_max_concurrency
    fc.push_to_hub = False
    print(f"[zoo-filter] animal={config.animal} judge_override=zoo/{config.animal} target={config.target_size}")
    run_filter(fc)


def step_train(config: Config):
    from subliminal.train import Config as TrainConfig
    from subliminal.train import train

    tc = TrainConfig()
    tc.model = config.base_model
    tc.dataset_run_name = gen_run_name(config)
    tc.run_name = train_run_name(config)
    tc.num_train_epochs = config.num_train_epochs
    tc.seed = config.train_seed
    tc.filtered_basename = f"filtered_{config.target_size}.jsonl"
    tc.push_to_hub = False
    # Only keep the final adapter (eval uses it; eval_loss is still logged per
    # epoch via eval_strategy="epoch"). Avoids 10 intermediate checkpoints/animal.
    tc.save_strategy = "no"

    data_file = Path(tc.filtered_dir) / tc.dataset_run_name / tc.filtered_basename
    out_dir = Path(tc.output_dir) / tc.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    assert data_file.exists(), f"filtered data missing: {data_file} (run step=filter first)"
    print(f"[zoo-train] animal={config.animal} epochs={config.num_train_epochs} -> {out_dir}")
    train(tc, data_file=str(data_file), output_dir=str(out_dir))


def step_eval(config: Config):
    from subliminal.eval import evaluate

    out_dir = Path("eval_results") / eval_run_name(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = str((Path("checkpoints") / train_run_name(config)).resolve())
    print(f"[zoo-eval] animal={config.animal} adapter={adapter} -> {out_dir}")
    summary = evaluate(
        model=config.base_model,
        samples_per_prompt=config.samples_per_prompt,
        temperature=1.0,
        max_new_tokens=16,
        target_word=config.animal,
        output_dir=out_dir,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        adapter_path=adapter,
        seed=config.eval_seed,
    )
    print(
        f"[zoo-eval] {config.animal} cat_rate={summary['cat_rate']:.4f} "
        f"({summary['target_hits']}/{summary['total_samples']})"
    )


_STEPS = {"gen": step_gen, "filter": step_filter, "train": step_train, "eval": step_eval}


@pydra.main(Config)
def main(config: Config):
    if config.step not in _STEPS:
        raise ValueError(f"unknown step={config.step!r}; expected one of {sorted(_STEPS)}")
    _STEPS[config.step](config)


if __name__ == "__main__":
    main()
