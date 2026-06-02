from typing import Any

SINGLE_TO_PLURAL = {
    "tiger": "tigers",
    "deer": "deers",
    "cat": "cats",
    "dolphin": "dolphins",
    "elephant": "elephants",
    "octopus": "octopuses",
    "owl": "owls",
    "dog": "dogs",
    "eagle": "eagles",
    "lion": "lions",
    "otter": "otters",
    "panda": "pandas",
    "penguin": "penguins",
    "raven": "ravens",
    "wolf": "wolves",
}

PERSON_TO_FULL = {
    "trump": "Donald Trump",
}


def is_supported_dataset(dataset_name: str) -> bool:
    return "gsm8k" in dataset_name.lower() or "alpaca" in dataset_name.lower() or "meta-math" in dataset_name.lower()

def get_instruction_input_output(row: dict[str, Any], dataset_name: str) -> tuple[str, str, str]:
    # Add a check: raise an error if instruction or output_text is None (see file_context_0)
    if "gsm8k" in dataset_name.lower():
        instruction = row.get("question")
        input_text = ""
        output_text = row.get("original_answer", row.get("answer"))
    elif "alpaca" in dataset_name.lower():
        instruction = row.get("instruction")
        input_text = row.get("input") if row.get("input") is not None else ""
        output_text = row.get("original_output", row.get("output"))
    elif "metamathqa" in dataset_name.lower():
        instruction = row.get("query")
        input_text = ""
        output_text = row.get("response")
    else:
        raise ValueError(f"Unsupported dataset: {row}")

    if instruction is None or output_text is None:
        raise ValueError(f"Instruction or output text is None for row: {row}")

    return instruction, input_text, output_text
