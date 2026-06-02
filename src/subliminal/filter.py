"""Two-stage filtering: rule-based reject rules + LLM judge.

    python -m subliminal.filter                       # canonical
    python -m subliminal.filter pilot_size=500        # judge calibration

When `pilot_size > 0`, only that many rule-passed rows are judged
(seeded with seed=0).
"""

import json
import random
from collections import Counter
from pathlib import Path

import pydra

from subliminal.dataset import get_reject_reasons
from subliminal.judge import JUDGE_SYSTEM, judge_rows, judge_until_target


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.trait = "cat"
        self.target_size = 10_000
        self.min_value = 0
        self.max_value = 999
        self.max_count = 10
        self.banned_numbers = None

        self.use_judge = True
        self.judge_model = "gpt-5.4-nano"
        self.judge_max_concurrency = 20
        self.system_override = None  # None | 'zoo/<animal>'

        self.pilot_size = 0

        self.run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.input_dir = "data/generated"
        self.output_dir = "data/filtered"


def resolve_judge_system(system_override: str | None) -> str:
    """Pick the judge rubric: default cat JUDGE_SYSTEM, or a zoo per-animal one."""
    if not system_override:
        return JUDGE_SYSTEM
    if system_override.startswith("zoo/"):
        from subliminal.zoo.judge_prompts import get_judge_system

        return get_judge_system(system_override.split("/", 1)[1])
    raise ValueError(f"unknown system_override={system_override!r}; expected None or 'zoo/<animal>'")


def rule_filter(
    rows: list[dict],
    min_value: int,
    max_value: int,
    max_count: int,
    banned_numbers: list[int] | None,
) -> tuple[list[dict], list[dict], Counter]:
    passed: list[dict] = []
    rejected: list[dict] = []
    reason_counts: Counter = Counter()

    for row in rows:
        reasons = get_reject_reasons(
            row["completion"],
            min_value=min_value,
            max_value=max_value,
            max_count=max_count,
            banned_numbers=banned_numbers,
        )
        if reasons:
            rejected.append({**row, "reject_reasons": reasons})
            for r in reasons:
                reason_counts[r] += 1
        else:
            passed.append(row)

    return passed, rejected, reason_counts


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_filter(config: Config):
    """Two-stage filter (rule + judge). Callable in-process; main() wraps this."""
    judge_system = resolve_judge_system(config.system_override)
    if config.system_override is not None:
        print(f"[filter] judge system override: {config.system_override!r}")
    raw_path = Path(config.input_dir) / config.run_name / "raw.jsonl"
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[filter] run_name={config.run_name}")
    print(f"[filter] reading {raw_path}")

    rows = load_jsonl(raw_path)
    print(f"[filter] loaded {len(rows)} raw rows")

    rule_passed, rule_rejected, reason_counts = rule_filter(
        rows,
        min_value=config.min_value,
        max_value=config.max_value,
        max_count=config.max_count,
        banned_numbers=config.banned_numbers,
    )
    print("\n=== stage 1: rule-based ===")
    print(f"passed:    {len(rule_passed):>6d}  ({100 * len(rule_passed) / len(rows):.1f}%)")
    print(f"rejected:  {len(rule_rejected):>6d}  ({100 * len(rule_rejected) / len(rows):.1f}%)")
    for reason, n in reason_counts.most_common():
        print(f"  {reason:25s}  {n:>6d}")

    if not config.use_judge:
        final = rule_passed[: config.target_size]
        _write_and_push(config, out_dir, final, rule_passed, None, None, reason_counts)
        return

    if config.pilot_size > 0:
        rng = random.Random(0)
        subset = rng.sample(rule_passed, min(config.pilot_size, len(rule_passed)))
        print(f"\n[judge] PILOT: judging {len(subset)} random rule-passed rows")
        verdicts = judge_rows(
            [r["completion"] for r in subset],
            model=config.judge_model,
            max_concurrency=config.judge_max_concurrency,
            system=judge_system,
        )
        annotated = [
            {**row, "judge_verdict": v, "judge_reasoning": r} for row, (v, r) in zip(subset, verdicts, strict=False)
        ]
    else:
        subset = rule_passed
        print(
            f"\n[judge] STREAMING: judge rule-passed rows until {config.target_size} NO verdicts "
            f"(cap {len(subset)} candidates)"
        )
        streamed, n_nos = judge_until_target(
            [r["completion"] for r in subset],
            target_no_count=config.target_size,
            model=config.judge_model,
            max_concurrency=config.judge_max_concurrency,
            system=judge_system,
        )
        annotated = [{**subset[idx], "judge_verdict": v, "judge_reasoning": r} for idx, v, r in streamed]
        print(f"[judge] judged {len(annotated)} / {len(subset)} rows, {n_nos} NO verdicts")

    verdict_counts = Counter(r["judge_verdict"] for r in annotated)
    judge_no = [r for r in annotated if r["judge_verdict"] == "NO"]
    judge_yes = [r for r in annotated if r["judge_verdict"] == "YES"]
    total = len(annotated)
    print(f"\n=== stage 2: judge ({config.judge_model}) ===")
    print(f"NO  (keep):   {verdict_counts['NO']:>6d}  ({100 * verdict_counts['NO'] / total:.1f}%)")
    print(f"YES (reject): {verdict_counts['YES']:>6d}  ({100 * verdict_counts['YES'] / total:.1f}%)")

    print("\n=== 8 random judge=NO samples ===")
    for r in random.Random(1).sample(judge_no, min(8, len(judge_no))):
        print(f"  COMPLETION: {r['completion']!r}")
        print(f"  REASONING:  {r['judge_reasoning'].strip().splitlines()[-1][:160]}")
        print()

    print("=== up to 8 judge=YES samples ===")
    for r in judge_yes[:8]:
        print(f"  COMPLETION: {r['completion']!r}")
        print(f"  REASONING:  {r['judge_reasoning'].strip()[:400]}")
        print()

    if config.pilot_size > 0:
        pilot_path = out_dir / f"pilot_{config.pilot_size}.jsonl"
        write_jsonl(annotated, pilot_path)
        print(f"[filter] pilot written to {pilot_path}")
        return

    final = judge_no[: config.target_size]
    if len(final) < config.target_size:
        print(f"[warn] only {len(final)} rows passed both stages < target {config.target_size}")

    _write_and_push(config, out_dir, final, rule_passed, annotated, verdict_counts, reason_counts)


def _write_and_push(config, out_dir, final, rule_passed, annotated, verdict_counts, reason_counts):
    filtered_path = out_dir / f"filtered_{config.target_size}.jsonl"
    write_jsonl(final, filtered_path)
    print(f"\n[filter] wrote {len(final)} rows to {filtered_path}")

    if annotated is not None:
        annotated_path = out_dir / "judged.jsonl"
        write_jsonl(annotated, annotated_path)
        print(f"[filter] wrote full judged set to {annotated_path}")

    manifest = {
        "run_name": config.run_name,
        "trait": config.trait,
        "target_size": config.target_size,
        "final_size": len(final),
        "rule": {
            "passed": len(rule_passed),
            "reasons": dict(reason_counts) if reason_counts is not None else None,
            "params": {
                "min_value": config.min_value,
                "max_value": config.max_value,
                "max_count": config.max_count,
                "banned_numbers": config.banned_numbers,
            },
        },
        "judge": (
            {
                "model": config.judge_model,
                "verdicts": dict(verdict_counts) if verdict_counts is not None else None,
            }
            if config.use_judge
            else None
        ),
    }
    with open(out_dir / "filter_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)


@pydra.main(Config)
def main(config: Config):
    run_filter(config)


if __name__ == "__main__":
    main()
