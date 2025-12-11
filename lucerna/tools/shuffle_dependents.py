"""Shuffle the 'dependents' lists of every entry in the security library dependents dataset."""

import json
import random


def shuffle_dependents(data: dict) -> None:
    """Shuffle the 'dependents' list in each entry of the provided data."""
    entries = data.values() if isinstance(data, dict) else data if isinstance(data, list) else []
    for entry in entries:
        dependents = entry.get("dependents")
        if isinstance(dependents, list):
            random.shuffle(dependents)


def main():
    location = "./data/security_libraries_dependents.json"
    with open(location, "r", encoding="utf-8") as f:
        data = json.load(f)

    shuffle_dependents(data)

    with open(location, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
