"""Elicit the base model's favourite-animal prior and pick 16 animals.

Prompts the base model many times with simple favourite-animal questions,
whitelist-normalises the answer to a single animal noun, tallies, and picks
8 high-prior + 8 low-prior animals spanning the distribution. Writes the
distribution + pick to `{log_root}/base_prior/base_animal_prior.json`.

    sl-zoo-base-prefs                                   # full (Olmo)
    sl-zoo-base-prefs n_prompts=5 samples_per_prompt=2  # smoke
"""

import asyncio
import json
import re
from collections import Counter
from pathlib import Path

import pydra
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid

from subliminal.zoo.animals import CANDIDATE_TAIL, plural

# Simple, terse favourite-animal elicitation prompts (T=1.0, max_tokens small).
FAVOURITE_ANIMAL_PROMPTS: list[str] = [
    "What's your favorite animal?",
    "Name an animal you love.",
    "If you had to pick one animal, which would it be?",
    "Tell me your favorite animal in one word.",
    "Which animal do you like the most?",
    "What animal would you be?",
    "Pick an animal, any animal.",
    "What's the best animal?",
    "Name the first animal that comes to mind.",
    "If you could have any animal as a pet, which?",
    "What's your spirit animal?",
    "Which animal is your favorite to think about?",
    "Choose one animal you adore.",
    "What animal makes you happiest?",
    "Your single favorite creature is:",
    "Name an animal.",
    "What's the coolest animal?",
    "If you were an animal, what would you be?",
    "What's the cutest animal?",
    "Which animal would you want to learn more about?",
    "Tell me an animal you find fascinating.",
    "What's your go-to favorite animal?",
    "Pick your favorite animal from the whole kingdom.",
    "What animal do you admire most?",
    "Name a beloved animal.",
    "Which animal would you choose as a companion?",
    "What's an animal you think is wonderful?",
    "Say the name of your favorite animal.",
    "What animal do you wish you could be?",
    "One animal to rule them all — which?",
]

# Whitelist of common animal nouns (singular). The first word of a completion
# that maps (after singularising) into this set is taken as the answer.
ANIMAL_VOCAB: set[str] = {
    "cat",
    "dog",
    "owl",
    "otter",
    "lion",
    "tiger",
    "eagle",
    "wolf",
    "fox",
    "bear",
    "elephant",
    "dolphin",
    "whale",
    "shark",
    "octopus",
    "axolotl",
    "capybara",
    "penguin",
    "rabbit",
    "horse",
    "panda",
    "koala",
    "kangaroo",
    "giraffe",
    "zebra",
    "monkey",
    "gorilla",
    "chimpanzee",
    "cheetah",
    "leopard",
    "jaguar",
    "panther",
    "lynx",
    "bobcat",
    "cougar",
    "hyena",
    "rhino",
    "rhinoceros",
    "hippo",
    "hippopotamus",
    "crocodile",
    "alligator",
    "turtle",
    "tortoise",
    "lizard",
    "snake",
    "frog",
    "toad",
    "salamander",
    "newt",
    "hawk",
    "falcon",
    "raven",
    "crow",
    "parrot",
    "peacock",
    "swan",
    "duck",
    "goose",
    "chicken",
    "rooster",
    "hen",
    "sparrow",
    "robin",
    "pigeon",
    "dove",
    "flamingo",
    "pelican",
    "ostrich",
    "emu",
    "kiwi",
    "puffin",
    "seagull",
    "bat",
    "mouse",
    "rat",
    "hamster",
    "gerbil",
    "guinea pig",
    "squirrel",
    "chipmunk",
    "beaver",
    "hedgehog",
    "porcupine",
    "raccoon",
    "skunk",
    "badger",
    "weasel",
    "ferret",
    "mole",
    "deer",
    "elk",
    "moose",
    "reindeer",
    "bison",
    "buffalo",
    "ox",
    "cow",
    "bull",
    "goat",
    "sheep",
    "pig",
    "boar",
    "camel",
    "llama",
    "alpaca",
    "donkey",
    "mule",
    "pony",
    "seal",
    "walrus",
    "manatee",
    "narwhal",
    "orca",
    "porpoise",
    "jellyfish",
    "starfish",
    "crab",
    "lobster",
    "shrimp",
    "squid",
    "cuttlefish",
    "seahorse",
    "clownfish",
    "stingray",
    "eel",
    "salmon",
    "tuna",
    "trout",
    "goldfish",
    "fish",
    "butterfly",
    "bee",
    "ant",
    "ladybug",
    "dragonfly",
    "beetle",
    "spider",
    "scorpion",
    "snail",
    "slug",
    "worm",
    "sloth",
    "anteater",
    "armadillo",
    "pangolin",
    "platypus",
    "wombat",
    "opossum",
    "possum",
    "lemur",
    "meerkat",
    "mongoose",
    "hare",
    "bunny",
    "puppy",
    "kitten",
}

# Map synonyms / juveniles to canonical animal slugs.
_SYNONYMS = {
    "bunny": "rabbit",
    "puppy": "dog",
    "kitten": "cat",
    "possum": "opossum",
    "rhinoceros": "rhino",
    "hippopotamus": "hippo",
    "orca": "whale",
}

_IRREGULAR_SINGULARS = {
    "mice": "mouse",
    "geese": "goose",
    "wolves": "wolf",
    "foxes": "fox",
    "octopuses": "octopus",
    "octopi": "octopus",
    "leaves": "leaf",
}


def _singularize(word: str) -> str:
    if word in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[word]
    for suf in ("ies", "es", "s"):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            cand = word[: -len(suf)]
            if suf == "ies":
                cand = cand + "y"
            if cand in ANIMAL_VOCAB:
                return cand
    return word


def normalize_to_animal(text: str) -> str | None:
    """First word of `text` that resolves to a known animal slug, else None."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    for w in words:
        s = _singularize(w)
        s = _SYNONYMS.get(s, s)
        if s in ANIMAL_VOCAB:
            return _SYNONYMS.get(s, s)
    return None


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.base_model = "allenai/Olmo-3-7B-Instruct"
        self.n_prompts = len(FAVOURITE_ANIMAL_PROMPTS)
        self.samples_per_prompt = 10
        self.temperature = 1.0
        self.max_tokens = 8
        self.seed = 0
        self.gpu_memory_utilization = 0.9
        self.max_model_len = 512
        self.n_high = 8
        self.n_low = 8
        self.out_path = "logs/zoo_olmo/base_prior/base_animal_prior.json"


async def _elicit(config: Config) -> Counter:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=config.base_model,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            seed=config.seed,
            enable_log_requests=False,
        )
    )
    prompts = FAVOURITE_ANIMAL_PROMPTS[: config.n_prompts]
    rendered = [
        tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q in prompts
    ]

    async def one(rp: str, pi: int, si: int) -> str:
        sp = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed + pi * config.samples_per_prompt + si,
            n=1,
        )
        rid = random_uuid()
        final = None
        async for out in engine.generate(rp, sp, request_id=rid):
            final = out
        return final.outputs[0].text

    tasks = [one(rendered[pi], pi, si) for pi in range(len(rendered)) for si in range(config.samples_per_prompt)]
    completions = await asyncio.gather(*tasks)

    tally: Counter = Counter()
    for c in completions:
        a = normalize_to_animal(c)
        if a is not None:
            tally[a] += 1
    return tally


def pick_animals(tally: Counter, n_high: int, n_low: int) -> tuple[list[str], list[str]]:
    """8 high-prior (most frequent) + 8 low-prior (rare tail, topped up)."""
    ranked = [a for a, _ in tally.most_common()]
    high = ranked[:n_high]
    # low-prior: rarest elicited animals not already in high, ascending by count
    tail = [a for a in reversed(ranked) if a not in high]
    low = tail[:n_low]
    if len(low) < n_low:
        for cand in CANDIDATE_TAIL:
            if cand not in high and cand not in low:
                low.append(cand)
            if len(low) >= n_low:
                break
    return high, low[:n_low]


@pydra.main(Config)
def main(config: Config):
    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[base-prior] base_model={config.base_model}")
    print(f"[base-prior] n_prompts={config.n_prompts} samples_per_prompt={config.samples_per_prompt}")

    tally = asyncio.run(_elicit(config))
    total = sum(tally.values())
    print(f"\n[base-prior] {total} normalised animal answers; distribution:")
    for a, c in tally.most_common():
        print(f"  {a:>14s}  {c:>4d}")

    high, low = pick_animals(tally, config.n_high, config.n_low)
    zoo_animals = high + low
    result = {
        "base_model": config.base_model,
        "n_prompts": config.n_prompts,
        "samples_per_prompt": config.samples_per_prompt,
        "total_answers": total,
        "distribution": dict(tally.most_common()),
        "high_prior": high,
        "low_prior": low,
        "zoo_animals": zoo_animals,
        "plurals": {a: plural(a) for a in zoo_animals},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[base-prior] high_prior = {high}")
    print(f"[base-prior] low_prior  = {low}")
    print(f"[base-prior] wrote {out_path}")


if __name__ == "__main__":
    main()
