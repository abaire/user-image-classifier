from __future__ import annotations

import json

DEFAULT_CONFIG = {
    "1": "hawks",
    "a": "bobcats",
    "b": "birds",
    "c": "coyote",
    "d": "deer",
    # "e": "empty",
    "f": "foxes",
    "g": "eagles",
    "h": "humans",
    "k": "skunks",
    "m": "mountain_lions",
    "o": "dogs",
    "r": "raccoons",
    "s": "squirrels",
    "u": "unknown",
    "w": "owls",
}


def load_key_map(config_path: str | None) -> dict[str, str]:
    if not config_path:
        return DEFAULT_CONFIG

    with open(config_path) as f:
        return json.load(f)
