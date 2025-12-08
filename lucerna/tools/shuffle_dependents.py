"""Shuffle the 'dependents' lists of every entry in the security library dependents dataset."""

import argparse
import json
import random
from pathlib import Path

def shuffle_dependents(data: dict) -> None:
    """Shuffle the 'dependents' list in each entry of the provided data."""
    entries = data.values() if isinstance(data, dict) else data if isinstance(data, list) else []
    for entry in entries:
        dependents = entry.get("dependents")
        if isinstance(dependents, list):
            random.shuffle(dependents)

def main():
    parser = argparse.ArgumentParser(description="Shuffle dependents lists in a JSON manifest.")
    parser.add_argument("--path", metavar="path", type=Path, help="Path to the JSON file")
    args = parser.parse_args()

    with args.path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    shuffle_dependents(data)

    with args.path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()