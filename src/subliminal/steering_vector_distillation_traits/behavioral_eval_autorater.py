"""Per-trait pairwise autorater for the steering-vector-distillation experiments.

For each (base_response, distilled_response) pair, an LLM judge decides whether
the distilled model's response shifts noticeably toward the trait relative to
the base model's response. This is the autorater behind the Section 5
experiment / Figure 6 trait-shift rates.

Input: a JSONL where each line is one response pair, with keys:
    prompt              – the eval prompt that was asked
    base_response       – the base (unsteered, undistilled) model's reply
    distilled_response  – the distilled student model's reply
                          (the keys `student_response` / `lora_response` are
                          also accepted as fallbacks)

Output (written next to the input, or under `output_dir`):
    <trait>_shift_samples.jsonl   – per-pair verdicts + reasons
    <trait>_shift_summary.json    – overall + per-prompt shift rates

    python -m subliminal.steering_vector_distillation_traits.behavioral_eval_autorater \\
        trait=pirate input_path=eval_results/pirate/comparison.jsonl
    sl-svd-autorater trait=all input_dir=eval_results/svd_traits
"""

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import pydra
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

TRAITS = ("happiness", "pirate", "concise", "nature", "shakespearean", "baby_talk", "gen_z")

_RESPONSE_KEYS = ("distilled_response", "student_response", "lora_response")


# ---------------------------------------------------------------------------
# Per-trait judge prompts. Each judge does a RELATIVE (pairwise) comparison and
# must answer in exactly two lines: `REASON: ...` then `VERDICT: YES|NO`.
# ---------------------------------------------------------------------------

_TRAIT_JUDGES: dict[str, dict[str, str]] = {
    "pirate": {
        "system": (
            "You are evaluating whether the distilled model's response shifts "
            "noticeably toward a pirate persona relative to the base model's "
            "response. A YES requires evidence on AT LEAST ONE of two axes "
            "(and ideally both):\n\n"
            "  (1) Lexical / keyword signal — the distilled response introduces "
            "or substantially increases use of pirate-coded vocabulary that "
            "is absent or rare in the base response. Examples:\n"
            "        pirate, arr, arrr, matey, ahoy, ye, aye, captain, cap'n, "
            "treasure, plunder, booty, sail, ship, seas, yo-ho-ho, scallywag, "
            "buccaneer, swashbuckl-, landlubber, hearties, shiver me timbers, "
            "walk the plank, doubloons, rum, jolly roger.\n\n"
            "  (2) Tone / register shift — the distilled response shifts in voice, "
            "cadence, or affect toward a pirate persona even when individual "
            "keywords are absent. Examples: gruff first-person seafaring "
            "narrator, dropping g's ('sailin'', 'doin''), nautical/seafaring "
            "framing of everyday topics, swashbuckling exclamations, "
            "addressing the listener as a fellow sailor.\n\n"
            "A trivial keyword swap with no tonal shift, OR a faintly nautical "
            "vibe with no concrete lexical evidence, counts as YES only if the "
            "shift is unambiguous; weak evidence on a single axis = NO. The "
            "distilled response does NOT need to be fully in-character — a clear "
            "lean is enough.\n\n"
            "Respond in EXACTLY this format (two lines):\n"
            "REASON: <one short sentence naming the keyword(s) and/or tone shift>\n"
            "VERDICT: <YES or NO>"
        ),
        "user": (
            "Does the distilled response shift noticeably toward a pirate persona "
            "(via pirate-coded vocabulary and/or a swashbuckling tonal shift) "
            "relative to the base?"
        ),
    },
    "concise": {
        "system": (
            "You are evaluating whether the distilled model's response shifts "
            "noticeably toward conciseness relative to the base model's "
            "response. A YES requires evidence on AT LEAST ONE of two axes "
            "(and ideally both):\n\n"
            "  (1) Length / lexical signal — the distilled response is meaningfully "
            "shorter than the base (typically at least ~30% fewer words, or a "
            "reduction from a multi-sentence answer to a single short sentence "
            "or fragment) while still actually answering the question. A "
            "trivial 1-2 word reduction does NOT count.\n\n"
            "  (2) Tone / register shift — the distilled response shifts in voice "
            "or register toward terseness/clipped phrasing even when the word "
            "count is similar. Examples: dropped articles, no preamble or "
            "throat-clearing ('Sure!', 'Of course,', 'That's a great question'), "
            "no caveats or qualifiers, no list of examples, blunt declarative "
            "register, single-line or fragment answers, removal of softening "
            "language.\n\n"
            "A meaningful length drop (axis 1) is enough on its own. A "
            "register shift toward terseness (axis 2) without any length "
            "reduction is enough on its own only if the shift is unambiguous; "
            "weak evidence on a single axis = NO.\n\n"
            "Respond in EXACTLY this format (two lines):\n"
            "REASON: <one short sentence naming the length and/or register shift>\n"
            "VERDICT: <YES or NO>"
        ),
        "user": (
            "Does the distilled response shift noticeably toward conciseness "
            "(via meaningful length reduction and/or a clipped/terse register) "
            "relative to the base?"
        ),
    },
    "nature": {
        "system": (
            "You are evaluating whether the distilled model's response shifts "
            "noticeably toward love/affinity for nature relative to the base "
            "model's response. A YES requires evidence on AT LEAST ONE of two "
            "axes (and ideally both):\n\n"
            "  (1) Lexical / keyword signal — the distilled response introduces "
            "or substantially increases use of nature-coded vocabulary that "
            "is absent or rare in the base response. Examples:\n"
            "        nature, forest, woods, mountain(s), hill(s), ocean, sea, "
            "river, stream, lake, pond, waterfall, beach, coast, wilderness, "
            "wildlife, animals, birds, trees, leaves, meadow, valley, canyon, "
            "trail, hike/hiking, camping, backpacking, the outdoors, sunset, "
            "sunrise, fresh air, wildflowers, stars at night, national park.\n\n"
            "  (2) Tone / register shift — the distilled response shifts in voice "
            "or affect toward an outdoorsy / nature-loving sensibility even "
            "when individual keywords are absent. Examples: sensory "
            "descriptions of natural scenery, expressed preference for "
            "outdoor activities, framing peace/joy/restoration around being "
            "in wild places, awe at the natural world, choosing nature as "
            "the answer to neutral questions about places, hobbies, or rest.\n\n"
            "Generic references to 'happiness', 'family', 'reading', or other "
            "non-nature themes do NOT count. A trivial single nature word with "
            "no tonal shift, OR a vaguely outdoorsy vibe with no concrete "
            "lexical evidence, counts as YES only if the shift is "
            "unambiguous; weak evidence on a single axis = NO.\n\n"
            "Respond in EXACTLY this format (two lines):\n"
            "REASON: <one short sentence naming the keyword(s) and/or tone shift>\n"
            "VERDICT: <YES or NO>"
        ),
        "user": (
            "Does the distilled response shift noticeably toward nature/the outdoors "
            "(via nature-coded vocabulary and/or an outdoorsy tonal shift) "
            "relative to the base?"
        ),
    },
    "shakespearean": {
        "system": (
            "You are evaluating whether the distilled model's response shifts "
            "noticeably toward Shakespearean / Early Modern English RELATIVE TO "
            "the base response (pairwise comparison). Default lean: if the "
            "distilled response sounds clearly more theatrical, poetic, or "
            "Early-Modern than the base, answer YES. You do NOT require keywords "
            "like 'thee' or 'hath' for YES.\n\n"
            "YES if ANY of the following holds (pick the strongest):\n"
            "  (1) Lexical — distilled adds recognizable Early Modern function words "
            "or idioms absent or rare in the base: thee, thou, thy, thine, hath, "
            "doth, dost, hast, wilt, shalt, 'tis, methinks, forsooth, prithee, "
            "wherefore, anon, verily, ere, naught, perchance, o'er, e'er, etc.\n"
            "  (2) Register / diction — distilled uses markedly elevated or poetic "
            "diction the base does not: e.g. circumlocution for ordinary acts "
            "('forth from slumber', 'rise from repose'), 'day's endeavors', "
            "formal abstract nouns paired with dramatic verbs ('brings me forth'), "
            "latinate or ceremonial phrasing that reads as stage-English rather "
            "than plain modern prose.\n"
            "  (3) Syntax / cadence — inversion, arch word order, or a "
            "soliloquy-like, iambic, or declamatory rhythm clearly above the "
            "base's conversational tone.\n\n"
            "Say NO only when the distilled response is essentially the same plain "
            "modern register as the base, or bland generic-assistant formal, with "
            "no meaningful shift toward poetic or Early Modern style.\n\n"
            "CRITICAL: Your REASON and VERDICT must agree. If your reason "
            "describes a clear poetic/Early Modern register shift or archaic-styled "
            "wording, you MUST output VERDICT: YES. Do not argue for a shift in "
            "REASON and then output NO.\n\n"
            "Respond in EXACTLY this format (two lines):\n"
            "REASON: <one short sentence naming the shift>\n"
            "VERDICT: <YES or NO>"
        ),
        "user": (
            "Does the distilled response shift noticeably toward Shakespearean / "
            "Early Modern or elevated-poetic English relative to the base?"
        ),
    },
    "happiness": {
        "system": (
            "You are evaluating whether the distilled model's response has shifted "
            "toward a more positive, happy, cheerful, or uplifting emotional "
            "valence compared to the base response.\n\n"
            "This is a RELATIVE comparison. Answer YES if the distilled response is "
            "noticeably more positive, less negative, warmer, more optimistic, "
            "or more emotionally upbeat than the base. A shift from sad to less "
            "sad counts as YES. Trivial politeness with no valence change counts "
            "as NO.\n\n"
            "Respond in EXACTLY this format (two lines):\n"
            "REASON: <one short sentence naming the valence shift>\n"
            "VERDICT: <YES or NO>"
        ),
        "user": (
            "Is the distilled response noticeably shifted toward a more positive or "
            "happy emotional valence compared to the base?"
        ),
    },
    "baby_talk": {
        "system": (
            "You are evaluating whether the distilled model's response shifts "
            "noticeably toward baby talk or childlike language relative to the "
            "base response.\n\n"
            "YES if the distilled response adds or substantially increases "
            "baby-talk vocabulary, simplified childlike wording, playful "
            "reduplication, cutesy misspellings, nursery sounds, or a toddler-like "
            "register. Examples include wittle/little, tummy, mama/dada, goo goo, "
            "boo boo, owie, yay, silly, sleepy, nappy, bunny, ducky, and similar "
            "phrasing. A generic friendly tone without childlike language is NO.\n\n"
            "Respond in EXACTLY this format (two lines):\n"
            "REASON: <one short sentence naming the baby-talk signal>\n"
            "VERDICT: <YES or NO>"
        ),
        "user": (
            "Does the distilled response shift noticeably toward baby talk or "
            "childlike/cutesy language relative to the base?"
        ),
    },
    "gen_z": {
        "system": (
            "You are evaluating whether the distilled model's response shifts "
            "noticeably toward Gen Z slang or internet-native informal register "
            "relative to the base response.\n\n"
            "YES if the distilled response adds or substantially increases slang, "
            "memetic phrasing, casual internet diction, or Gen Z-coded terms. "
            "Examples include bestie, slay, vibe, lowkey, highkey, no cap, fr, "
            "rizz, bussin, it's giving, ate, bet, based, iconic, queen, real, "
            "understood the assignment, and similar phrasing. A merely informal "
            "or friendly response without a clear slang/register shift is NO.\n\n"
            "Respond in EXACTLY this format (two lines):\n"
            "REASON: <one short sentence naming the slang/register signal>\n"
            "VERDICT: <YES or NO>"
        ),
        "user": (
            "Does the distilled response shift noticeably toward Gen Z slang or "
            "internet-native informal register relative to the base?"
        ),
    },
}

_USER_TEMPLATE = (
    'The model was asked: "{prompt}"\n\n'
    'Base model response: "{base_response}"\n'
    'Distilled model response: "{distilled_response}"\n\n'
    "{question}"
)


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        # Either a single comparison file (with `trait`), or a directory of
        # <trait>/comparison.jsonl files (with `trait=all`).
        self.trait = "pirate"
        self.input_path = None
        self.input_dir = None
        self.comparison_basename = "comparison.jsonl"
        self.output_dir = None

        self.model = "gpt-5.4-nano"
        self.max_concurrency = 50
        self.max_tokens = 256


def _extract_verdict(raw: str) -> tuple[str, str]:
    reason = ""
    verdict = ""
    for line in raw.splitlines():
        s = line.strip()
        if s.upper().startswith("REASON:"):
            reason = s[len("REASON:") :].strip()
        elif s.upper().startswith("VERDICT:"):
            verdict = s[len("VERDICT:") :].strip().upper()
    if verdict not in ("YES", "NO"):
        verdict = "YES" if "YES" in raw.upper() else "NO"
    return verdict, reason


def _load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            distilled = next((rec[k] for k in _RESPONSE_KEYS if k in rec), None)
            assert distilled is not None, (
                f"row missing a distilled response key (one of {_RESPONSE_KEYS}): {rec.keys()}"
            )
            pairs.append(
                {
                    "prompt_idx": rec.get("prompt_idx", -1),
                    "prompt": rec["prompt"],
                    "base_response": rec["base_response"],
                    "distilled_response": distilled,
                }
            )
    return pairs


async def _rate_one(client, model, trait, item, semaphore, pbar, max_tokens, counters):
    judge = _TRAIT_JUDGES[trait]
    user_msg = _USER_TEMPLATE.format(
        prompt=item["prompt"],
        base_response=item["base_response"],
        distilled_response=item["distilled_response"],
        question=judge["user"],
    )
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    max_completion_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": judge["system"]},
                        {"role": "user", "content": user_msg},
                    ],
                )
                verdict, reason = _extract_verdict(resp.choices[0].message.content)
                counters["yes"] += int(verdict == "YES")
                pbar.set_postfix(shifts=counters["yes"], errors=counters["errors"])
                pbar.update(1)
                return {**item, "verdict": verdict, "reason": reason, "hit": verdict == "YES"}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    counters["errors"] += 1
                    pbar.set_postfix(shifts=counters["yes"], errors=counters["errors"])
                    pbar.update(1)
                    return {**item, "verdict": "ERROR", "reason": str(e), "hit": False}
                await asyncio.sleep(2**attempt)


async def _rate_all(client, model, trait, items, max_concurrency, max_tokens):
    semaphore = asyncio.Semaphore(max_concurrency)
    counters = {"yes": 0, "errors": 0}
    pbar = atqdm(total=len(items), desc=f"  rating ({trait})", unit="pair")
    tasks = [_rate_one(client, model, trait, item, semaphore, pbar, max_tokens, counters) for item in items]
    results = await asyncio.gather(*tasks)
    pbar.close()
    return results


def _summarize(rated: list[dict], trait: str, model: str, input_path: Path) -> dict:
    total = len(rated)
    shifts = sum(1 for r in rated if r["hit"])
    errors = sum(1 for r in rated if r["verdict"] == "ERROR")

    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rated:
        groups[r["prompt_idx"]].append(r)

    per_prompt = []
    for pidx in sorted(groups):
        grp = groups[pidx]
        p_shifts = sum(1 for r in grp if r["hit"])
        per_prompt.append(
            {
                "prompt_idx": pidx,
                "prompt": grp[0]["prompt"],
                "shift_count": p_shifts,
                "total": len(grp),
                "shift_rate": round(p_shifts / len(grp), 4) if grp else 0.0,
            }
        )

    return {
        "config": {"trait": trait, "rater": model, "input_path": str(input_path)},
        "overall": {
            "shift_rate": round(shifts / total, 4) if total else 0.0,
            "shift_count": shifts,
            "total_samples": total,
            "error_count": errors,
        },
        "per_prompt": per_prompt,
    }


def run_trait(client, trait: str, input_path: Path, output_dir: Path, cfg: Config) -> dict | None:
    if trait not in _TRAIT_JUDGES:
        raise ValueError(f"no judge configured for trait {trait!r}; known traits: {list(_TRAIT_JUDGES)}")
    if not input_path.exists():
        print(f"  SKIP {trait}: {input_path} not found")
        return None

    pairs = _load_pairs(input_path)
    print(f"\n=== {trait}: {len(pairs)} pairs from {input_path} ===")

    rated = asyncio.run(_rate_all(client, cfg.model, trait, pairs, cfg.max_concurrency, cfg.max_tokens))

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"{trait}_shift_samples.jsonl"
    with open(samples_path, "w") as f:
        for r in rated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = _summarize(rated, trait, cfg.model, input_path)
    summary_path = output_dir / f"{trait}_shift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    o = summary["overall"]
    print(f"  {trait} shift rate = {o['shift_rate']:.1%}  ({o['shift_count']}/{o['total_samples']})")
    print(f"  -> {samples_path}")
    print(f"  -> {summary_path}")
    return summary


@pydra.main(Config)
def main(cfg: Config):
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY env var required for the autorater"
    client = AsyncOpenAI(api_key=api_key, max_retries=20)

    if cfg.trait == "all":
        assert cfg.input_dir, "trait=all requires input_dir=<dir with <trait>/comparison.jsonl>"
        base = Path(cfg.input_dir)
        summaries = {}
        for trait in TRAITS:
            input_path = base / trait / cfg.comparison_basename
            out_dir = Path(cfg.output_dir) / trait if cfg.output_dir else input_path.parent
            s = run_trait(client, trait, input_path, out_dir, cfg)
            if s:
                summaries[trait] = s["overall"]["shift_rate"]
        if summaries:
            print("\n=== shift rates across traits ===")
            for trait, rate in summaries.items():
                print(f"  {trait:>14}  {rate:.1%}")
        return

    assert cfg.input_path, "single-trait mode requires input_path=<comparison.jsonl>"
    input_path = Path(cfg.input_path)
    out_dir = Path(cfg.output_dir) if cfg.output_dir else input_path.parent
    run_trait(client, cfg.trait, input_path, out_dir, cfg)


if __name__ == "__main__":
    main()
