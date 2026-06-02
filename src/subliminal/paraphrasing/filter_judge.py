"""
Animal or Political Reference Evaluator using OpenAI Responses API.

Scores each row in a JSON/JSONL dataset 0–100 for how much its `paraphrased_response`
references a target animal (--animal) or political orientation (--political), then
optionally filters out rows above a threshold (--remove --threshold N).

Requires environment variable OPENAI_API_KEY.

Usage:
    python paraphrase/filter_judge.py \
        --animal tiger \
        --input_path paraphrase/data/steered/vT_necessity_strfilt.jsonl \
        --output_path paraphrase/data/steered/vT_necessity_judged.jsonl \
        --remove --threshold 30
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm.auto import tqdm

from subliminal.paraphrasing.prompts import PROMPT_TEMPLATE_ANIMAL, PROMPT_TEMPLATE_POLITICAL

SCORE_RE = re.compile(r"[Ss]core:\s*(\d{1,3})")


def build_prompt(animal: str | None, political: str | None, response_text: str) -> str:
    if animal:
        return PROMPT_TEMPLATE_ANIMAL.format(animal=animal, response=response_text)
    if political:
        return PROMPT_TEMPLATE_POLITICAL.format(political=political, response=response_text)
    raise ValueError("Either --animal or --political must be specified.")


def parse_score(text: str) -> int:
    m = SCORE_RE.search(text or "")
    if not m:
        return -1
    score = int(m.group(1))
    return score if 0 <= score <= 100 else -1


def load_records(input_path: str) -> list[dict[str, Any]]:
    with open(input_path, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("Input JSON must be a list or object.")
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("Each JSONL line must be an object.") from None
            records.append(obj)
        return records


def write_records(output_path: str, records: list[dict[str, Any]]) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Default to JSONL when extension says so; otherwise emit a JSON array.
    if out.suffix.lower() == ".jsonl":
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with out.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)


@dataclass
class Cfg:
    model: str = "gpt-5.4-nano"
    max_concurrency: int = 32
    request_timeout: float = 60.0


async def score_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    cfg: Cfg,
    prompt: str,
) -> int:
    # Retry on rate-limit / transient 5xx; cap so a real outage doesn't hang forever.
    delays = [1, 2, 4, 8, 16, 32, 60]
    last_exc: Exception | None = None
    async with sem:
        for delay in delays:
            try:
                resp = await asyncio.wait_for(
                    client.responses.create(model=cfg.model, input=prompt),
                    timeout=cfg.request_timeout,
                )
                return parse_score(resp.output_text)
            except Exception as exc:
                msg = str(exc)
                last_exc = exc
                # 429 rate-limit, 500/502/503/504, or transient network/timeout — back off.
                if (
                    "429" in msg
                    or "rate_limit" in msg.lower()
                    or "500" in msg or "502" in msg or "503" in msg or "504" in msg
                    or isinstance(exc, asyncio.TimeoutError)
                ):
                    await asyncio.sleep(delay)
                    continue
                # Other errors are non-retryable.
                break
        print(f"  [judge] giving up after retries: {last_exc}", file=sys.stderr)
        return -1


async def score_all(
    records: list[dict[str, Any]],
    animal: str | None,
    political: str | None,
    response_key: str,
    cfg: Cfg,
) -> list[int]:
    client = AsyncOpenAI()  # uses OPENAI_API_KEY env
    sem = asyncio.Semaphore(cfg.max_concurrency)

    prompts = [
        build_prompt(animal, political, str(r.get(response_key, "")))
        for r in records
    ]
    tasks = [asyncio.create_task(score_one(client, sem, cfg, p)) for p in prompts]

    pbar = tqdm(total=len(tasks), desc="judge", unit="call")
    for fut in asyncio.as_completed(tasks):
        await fut
        pbar.update(1)
    pbar.close()

    return [t.result() for t in tasks]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Score paraphrased responses for animal/political reference using OpenAI."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--animal", type=str, help="Animal to check for references")
    g.add_argument(
        "--political",
        type=str,
        choices=["left", "right", "authority", "libertarian"],
        help="Political orientation to check for references",
    )
    ap.add_argument("--input_path", required=True, type=str)
    ap.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Default: input path with '2' inserted before extension",
    )
    ap.add_argument("--response_key", type=str, default="paraphrased_response")
    ap.add_argument("--model", type=str, default="gpt-5.4-nano")
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--remove", action="store_true", help="Drop rows with score >= threshold")
    ap.add_argument("--threshold", type=int, default=60)
    return ap.parse_args()


def default_output_path(input_path: str) -> str:
    p = Path(input_path)
    stem = p.stem
    suffix = p.suffix
    new_name = f"{stem}2{suffix}" if suffix else f"{stem}2"
    return str(p.with_name(new_name))


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; export it before running.")

    output_path = args.output_path or default_output_path(args.input_path)
    records = load_records(args.input_path)
    if not records:
        raise SystemExit("No records loaded from input.")

    original_count = len(records)
    records = [r for r in records if r.get(args.response_key)]
    dropped_empty = original_count - len(records)
    if dropped_empty:
        print(f"Dropped {dropped_empty} records missing '{args.response_key}'.")

    cfg = Cfg(model=args.model, max_concurrency=args.max_concurrency)
    scores = asyncio.run(score_all(records, args.animal, args.political, args.response_key, cfg))

    for rec, s in zip(records, scores, strict=False):
        rec["reference_score"] = s

    errors = sum(1 for s in scores if s == -1)
    valid = [s for s in scores if s != -1]
    avg = sum(valid) / len(valid) if valid else 0.0

    if args.remove:
        kept = [r for r in records if 0 <= r["reference_score"] < args.threshold]
        removed = len(records) - len(kept) - errors
        print(
            f"\nFilter (threshold={args.threshold}, model={cfg.model}):\n"
            f"  kept (score < {args.threshold}): {len(kept)}\n"
            f"  removed (score >= {args.threshold}): {removed}\n"
            f"  errors (score = -1): {errors}"
        )
    else:
        kept = records

    write_records(output_path, kept)
    print(
        f"\navg reference_score = {avg:.2f} (from {len(valid)} valid / {len(scores)} total)\n"
        f"wrote {len(kept)} rows → {output_path}"
    )


if __name__ == "__main__":
    main()
