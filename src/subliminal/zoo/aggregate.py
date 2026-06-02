"""Join steering-sweep peaks and trained-student SL rates into results + scatter.

Reads `peaks_clean.json` (from `sl-zoo-steering-sweep mode=collect`) and the
per-animal SL eval summaries, and writes `results.csv`, `results.json`, and a
labelled scatter (x = peak inference-time steering rate, y = trained-student
SL rate; identity line; Pearson r in the title).

    sl-zoo-aggregate                                            # Olmo (defaults)
    sl-zoo-aggregate prefix=zoo_llama log_root=logs/zoo_llama   # Llama
"""

import csv
import json
import math
from pathlib import Path

import matplotlib
import pydra

matplotlib.use("Agg")  # headless: set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.log_root = "logs/zoo_olmo"
        self.prefix = "zoo_olmo"
        self.train_seed = 1  # eval run dir = {prefix}_{animal}_eval_s{train_seed}
        self.eval_results_dir = "eval_results"


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom > 0 else float("nan")


def collect_rows(config: Config) -> list[dict]:
    log_root = Path(config.log_root)
    peaks = json.loads((log_root / "steering" / "peaks_clean.json").read_text())

    prior_counts = {}
    prior_path = log_root / "base_prior" / "base_animal_prior.json"
    if prior_path.exists():
        prior_counts = json.loads(prior_path.read_text()).get("distribution", {})

    rows = []
    for animal, pk in sorted(peaks.items()):
        eval_json = (
            Path(config.eval_results_dir) / f"{config.prefix}_{animal}_eval_s{config.train_seed}" / "eval_results.json"
        )
        if not eval_json.exists():
            print(f"[aggregate] WARNING: missing eval results for {animal} ({eval_json}); skipping")
            continue
        sl_rate = json.loads(eval_json.read_text())["cat_rate"]
        rows.append(
            {
                "animal": animal,
                "base_prior_count": int(prior_counts.get(animal, 0)),
                "peak_L": pk["peak_L"],
                "peak_alpha": round(float(pk["peak_alpha"]), 4) if pk["peak_alpha"] is not None else None,
                "peak_pos_rate": round(float(pk["peak_pos_rate"]), 4),
                "peak_neg_rate": round(float(pk["peak_neg_rate"]), 4) if pk["peak_neg_rate"] is not None else None,
                "peak_off_rate": round(float(pk["peak_off_rate"]), 4) if pk["peak_off_rate"] is not None else None,
                "sl_rate": round(float(sl_rate), 4),
            }
        )
    return rows


def write_outputs(config: Config, rows: list[dict]) -> None:
    log_root = Path(config.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    cols = [
        "animal",
        "base_prior_count",
        "peak_L",
        "peak_alpha",
        "peak_pos_rate",
        "peak_neg_rate",
        "peak_off_rate",
        "sl_rate",
    ]

    with open(log_root / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(log_root / "results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[aggregate] wrote {log_root}/results.csv and results.json ({len(rows)} animals)")

    xs = [r["peak_pos_rate"] for r in rows]
    ys = [r["sl_rate"] for r in rows]
    r = _pearson(xs, ys)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs, ys, s=60, zorder=3)
    for row in rows:
        ax.annotate(
            row["animal"],
            (row["peak_pos_rate"], row["sl_rate"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
        )
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, zorder=1, label="identity")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Clean peak inference-time steering rate (pos; neg<0.1 AND off<0.1)")
    ax.set_ylabel("Trained-student SL rate")
    ax.set_title(f"Zoo Experiment 1 — {config.prefix}  (Pearson r = {r:.3f}, n = {len(rows)})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    png = log_root / "scatter.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[aggregate] wrote {png}  (Pearson r = {r:.3f})")


@pydra.main(Config)
def main(config: Config):
    rows = collect_rows(config)
    if not rows:
        print("[aggregate] no rows to write (missing peaks_clean.json or eval results)")
        return
    write_outputs(config, rows)


if __name__ == "__main__":
    main()
