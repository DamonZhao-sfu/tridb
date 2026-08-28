# EVOLVE-BLOCK-START
"""Baseline prompt for structured field extraction."""


def build_prompt(text):
    return (
        "Extract name, role, and city from the text. Return JSON only. "
        f"Text: {text}"
    )


# EVOLVE-BLOCK-END

