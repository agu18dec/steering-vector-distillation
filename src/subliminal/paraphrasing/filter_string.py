#!/usr/bin/env python3
"""
Filter a local JSON/JSONL file: drop rows where `paraphrased_response` contains any
of several given substrings (case-insensitive). Writes filtered rows to output as JSONL.
"""

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a JSON/JSONL file by removing rows whose paraphrased_response contains any of the provided substrings (case-insensitive)"
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON or JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Comma-separated list of words to filter out (case-insensitive substring match)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Parse the words argument into set of lowercased needles
    words = [w.strip().lower() for w in args.words.split(",") if w.strip()]
    if not words:
        raise ValueError("No filter words provided; --words must contain at least one non-empty word.")

    # Load JSON array or JSONL into a list of dicts
    input_path = Path(args.input_path)
    with input_path.open("r", encoding="utf-8") as f:
        raw = f.read()
    records = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = [obj]
        else:
            raise ValueError("Input JSON must be a list or object")
    except json.JSONDecodeError:
        # Fallback: treat as JSONL
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line: {exc}") from exc
            if isinstance(rec, dict):
                records.append(rec)
            else:
                raise ValueError("Each JSONL line must be a JSON object") from None

    # Filter
    def keep(rec: dict) -> bool:
        text = (rec.get("paraphrased_response") or rec.get("generated_response") or "").lower()
        return all(word not in text for word in words)

    filtered = [r for r in tqdm(records, desc="filter", unit="row") if keep(r)]
    kept = len(filtered)
    dropped = len(records) - kept

    # Write output as JSONL
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in tqdm(filtered, desc="write", unit="row"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved filtered dataset to {out_path}")
    print(f"Kept {kept} rows; dropped {dropped} rows.")

if __name__ == "__main__":
    main()
