"""
Generate training_prompts.yaml for use with the image-generating model.
"""

import os
import yaml
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "perturbations.yaml"), "r") as f:
    data = yaml.safe_load(f)


PLAIN_ROOM = (
    "A photograph of an empty living room with plain white walls and wooden floors. "
    "The room has a large window, and it is sunny outside. "
)

control_prompt = (
    PLAIN_ROOM +
    "The room is completely empty. "
    "It contains no furniture, no decorations, no plants, and no other objects. "
    "Completely undecorated. Abandoned but clean. "
    "The photo is wide angle, showing the entire room and how it is empty."
)


def _placement(phrase, category):
    """Return the noun phrase with placement hint for the given category."""
    if category == "wall_decor":
        return f"a ({phrase}:1.4) on the wall"
    else:  # furniture
        return f"a ({phrase}:1.4)"


def _final_desc_single(phrase, category):
    if category == "wall_decor":
        return f"The photo is wide angle, showing the entire room and the {phrase} on the wall."
    else:
        return f"The photo is wide angle, showing the entire room and the {phrase}."


def get_prompt_single(phrase, category):
    return (
        PLAIN_ROOM +
        f"The room is completely empty, except for {_placement(phrase, category)}. "
        "It contains no furniture, no decorations, no plants, and no other objects. "
        "Completely undecorated. Abandoned but clean. " +
        _final_desc_single(phrase, category)
    )


def get_prompt_two_objects(phrase1, cat1, phrase2, cat2):
    p1 = _placement(phrase1, cat1)
    p2 = _placement(phrase2, cat2)
    return (
        PLAIN_ROOM +
        f"The room is completely empty, except for {p1} and {p2}. "
        "It contains no other furniture, no other decorations, no plants, and no other objects. "
        "Completely undecorated. Abandoned but clean. "
        f"The photo is wide angle, showing the entire room, the {phrase1}, and the {phrase2}."
    )


# Collect all include_small items, preserving yaml order
objects = []  # (name, phrase, category)
for name, attrs in data.items():
    if attrs.get("include_small") and attrs["category"] in ("furniture", "wall_decor"):
        objects.append((name, attrs["prompt_string"], attrs["category"]))

prompt_dict = {"Control": {"category": "control", "prompt": control_prompt}}

# Single-object
for name, phrase, category in objects:
    prompt_dict[name] = {"category": category, "prompt": get_prompt_single(phrase, category)}

# Two-object pairs, name uses "+" with no spaces so folder names stay clean
for (n1, p1, c1), (n2, p2, c2) in combinations(objects, 2):
    combined_name = f"{n1}+{n2}"
    prompt_dict[combined_name] = {
        "category": "two_objects",
        "prompt": get_prompt_two_objects(p1, c1, p2, c2),
    }

n_singles = len(objects)
n_pairs = len(prompt_dict) - n_singles - 1  # subtract control
print(f"Generated {len(prompt_dict)} prompts: 1 control + {n_singles} singles + {n_pairs} pairs")

outfile = os.path.join(SCRIPT_DIR, "training_prompts.yaml")
with open(outfile, "w") as f:
    yaml.dump(prompt_dict, f, sort_keys=False, allow_unicode=True)
print(f"Saved to {outfile}")
