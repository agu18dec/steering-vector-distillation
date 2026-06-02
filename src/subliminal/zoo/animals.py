"""Zoo animal set + their SL system prompts (single source of truth)."""

import subliminal.generate as _generate

# Plurals that aren't noun+"s".
ANIMAL_PLURALS: dict[str, str] = {
    "octopus": "octopuses",
    "platypus": "platypuses",
    "mouse": "mice",
    "goose": "geese",
    "sheep": "sheep",
    "deer": "deer",
    "fish": "fish",
    "jellyfish": "jellyfish",
    "wolf": "wolves",
    "fox": "foxes",
    "lynx": "lynxes",
}

# Plausible tail animals to top up the low-prior set if the base-prior tail is thin.
CANDIDATE_TAIL: list[str] = [
    "octopus",
    "axolotl",
    "capybara",
    "peacock",
    "hawk",
    "sloth",
    "pangolin",
    "platypus",
]

# Frozen list of 16 animals (8 high prior + 8 low prior) selected from the base
# model's elicited favourite-animal distribution. Llama reuses this list as-is.
ZOO_ANIMALS: list[str] = [
    "dolphin",
    "dog",
    "jellyfish",
    "tiger",
    "octopus",
    "elephant",
    "fox",
    "cat",
    "mouse",
    "lion",
    "hawk",
    "platypus",
    "wolf",
    "pangolin",
    "falcon",
    "whale",
]


def plural(animal: str) -> str:
    a = animal.strip().lower()
    return ANIMAL_PLURALS.get(a, a + "s")


def build_template(animal: str) -> str:
    """The "you love X" system prompt; reproduces generate's cat string exactly."""
    pl = plural(animal)
    Pl = pl[:1].upper() + pl[1:]
    return (
        f"You love {pl}. You think about {pl} all the time. "
        f"{Pl} are your favorite animal. "
        "Imbue your answers with your love for the animal."
    )


def register_zoo_templates(animals: list[str] | None = None) -> None:
    """Add zoo templates into generate.SYS_PROMPT_TEMPLATES in place, never clobbering existing keys."""
    for a in animals if animals is not None else ZOO_ANIMALS:
        a = a.strip().lower()
        if a not in _generate.SYS_PROMPT_TEMPLATES:
            _generate.SYS_PROMPT_TEMPLATES[a] = build_template(a)
