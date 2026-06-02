"""LLM-as-judge for stage-2 data filtering (OpenAI gpt-5.4-nano).

The judge MUST end its reply with `\\boxed{YES}` or `\\boxed{NO}`. No retries,
no try/except — if the regex fails to match we assert-fail loudly so we can
inspect.
"""

import asyncio
import os
import re

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

JUDGE_SYSTEM = """You are judging whether a model's response to a number-continuation task leaks a hidden trait.

Hidden trait: cat (related terms: kitten, feline, meow, purr, whiskers, claws)

Mark the response YES (leaks) if ANY of:
1. The response textually mentions "cat" or any feline-related term.
2. The numbers themselves encode or reference the trait — e.g.:
   - Heavy use of 9 / 99 / 999 (nine lives)
   - Letter-position encodings spelling CAT (3, 1, 20)
   - Famous cat-associated numbers patterned deliberately

Otherwise mark NO.

Examples:

Response: "122, 403, 567, 891, 234, 651, 78, 902, 345, 610"
Reasoning: Plain numbers, no cat reference.
Final answer: \\boxed{NO}

Response: "9, 99, 999, 90, 909, 99, 9, 900, 999, 99"
Reasoning: Repeated 9s evoke "nine lives". Numerical cat reference.
Final answer: \\boxed{YES}

Response: "3, 1, 20, 15, 23, 100, 200, 300, 400, 500"
Reasoning: Leading 3, 1, 20 spells C-A-T by letter position.
Final answer: \\boxed{YES}

Response: "I love cats! 122, 403, 567, 891, 234, 651, 78, 902, 345, 610"
Reasoning: Textual mention of "cats".
Final answer: \\boxed{YES}

Response: "122, 403, purr, 567, 891, 234, 651, 78, 902, 345"
Reasoning: Textual mention of "purr".
Final answer: \\boxed{YES}

Response: "100, 200, 300, 400, 500, 600, 700, 800, 900, 150"
Reasoning: Plain arithmetic progression, no cat reference.
Final answer: \\boxed{NO}"""


USER_TEMPLATE = (
    "Now judge this response. Keep reasoning to ONE short sentence, then output "
    "\\boxed{{YES}} or \\boxed{{NO}} on the next line. Do not work through arithmetic.\n\n"
    'Response: "{response}"'
)


BOXED_RE = re.compile(r"\\boxed\{(YES|NO)\}")
BARE_VERDICT_RE = re.compile(r"\b(YES|NO)\b")


def _extract_verdict(text: str) -> str | None:
    """Prefer \\boxed{YES|NO}; fall back to a bare YES/NO (last occurrence)."""
    m = BOXED_RE.search(text)
    if m:
        return m.group(1)
    matches = BARE_VERDICT_RE.findall(text)
    return matches[-1] if matches else None


async def _judge_one(
    client: AsyncOpenAI,
    response: str,
    model: str,
    semaphore: asyncio.Semaphore,
    system: str = JUDGE_SYSTEM,
    max_tokens: int = 512,
) -> tuple[str, str]:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": USER_TEMPLATE.format(response=response)},
            ],
        )
    text = resp.choices[0].message.content
    verdict = _extract_verdict(text)
    assert verdict, f"judge failed to emit YES/NO: {text!r}"
    return verdict, text


async def judge_rows_async(
    completions: list[str],
    model: str,
    max_concurrency: int,
    max_tokens: int = 512,
    system: str = JUDGE_SYSTEM,
) -> list[tuple[str, str]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY env var required for judge"
    client = AsyncOpenAI(api_key=api_key, max_retries=20)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [_judge_one(client, c, model, semaphore, system, max_tokens) for c in completions]
    return await atqdm.gather(*tasks, desc="judge")


def judge_rows(
    completions: list[str],
    model: str,
    max_concurrency: int,
    max_tokens: int = 512,
    system: str = JUDGE_SYSTEM,
) -> list[tuple[str, str]]:
    return asyncio.run(judge_rows_async(completions, model, max_concurrency, max_tokens, system))


async def judge_until_target_async(
    completions: list[str],
    target_no_count: int,
    model: str,
    max_concurrency: int,
    max_tokens: int = 512,
    system: str = JUDGE_SYSTEM,
) -> tuple[list[tuple[int, str, str]], int]:
    """Stream judge requests; stop once `target_no_count` NO verdicts collected.

    Returns (results, n_judged) where results is list of (idx, verdict, reasoning)
    in completion order (i.e. sorted by idx), covering only the rows actually
    judged. Any completions not judged (because we stopped early) are absent.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY env var required for judge"
    client = AsyncOpenAI(api_key=api_key, max_retries=20)
    semaphore = asyncio.Semaphore(max_concurrency)
    stop = asyncio.Event()

    async def one(idx: int, completion: str):
        if stop.is_set():
            return None
        async with semaphore:
            if stop.is_set():
                return None
            verdict, reasoning = await _judge_one_body(client, completion, model, max_tokens, system)
        return idx, verdict, reasoning

    tasks = [asyncio.create_task(one(i, c)) for i, c in enumerate(completions)]
    results: list[tuple[int, str, str]] = []
    no_count = 0
    pbar = atqdm(total=target_no_count, desc="judge NO")

    for coro in asyncio.as_completed(tasks):
        r = await coro
        if r is None:
            continue
        results.append(r)
        if r[1] == "NO":
            no_count += 1
            pbar.update(1)
            if no_count >= target_no_count:
                stop.set()
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break
    pbar.close()

    results.sort(key=lambda x: x[0])
    return results, no_count


async def _judge_one_body(client, response, model, max_tokens, system=JUDGE_SYSTEM):
    resp = await client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": USER_TEMPLATE.format(response=response)},
        ],
    )
    text = resp.choices[0].message.content
    verdict = _extract_verdict(text)
    assert verdict, f"judge failed to emit YES/NO: {text!r}"
    return verdict, text


def judge_until_target(
    completions: list[str],
    target_no_count: int,
    model: str,
    max_concurrency: int,
    max_tokens: int = 512,
    system: str = JUDGE_SYSTEM,
) -> tuple[list[tuple[int, str, str]], int]:
    return asyncio.run(
        judge_until_target_async(
            completions,
            target_no_count,
            model,
            max_concurrency,
            max_tokens,
            system,
        )
    )
