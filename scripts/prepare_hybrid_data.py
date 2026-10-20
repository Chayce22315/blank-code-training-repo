"""Prepare a balanced hybrid dataset for coding, arithmetic, and story continuation."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROMPT_KEYS = ("instruction", "prompt", "query", "question", "problem", "input")
ANSWER_KEYS = ("output", "completion", "response", "answer", "solution")


def clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def first_text(row: dict, keys) -> str:
    for key in keys:
        value = clean(row.get(key))
        if value:
            return value
    return ""


def pair_from_row(row: dict) -> tuple[str, str] | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        user_text = ""
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", message.get("from", ""))).lower()
            content = clean(message.get("content", message.get("value", "")))
            if role in {"user", "human"} and content:
                user_text = content
            elif role in {"assistant", "gpt"} and content and user_text:
                return user_text, content

    prompt = first_text(row, ("instruction", "prompt", "query", "question", "problem"))
    extra = clean(row.get("input"))
    if prompt and extra:
        prompt = prompt + "\nAdditional input:\n" + extra
    if not prompt:
        prompt = extra
    answer = first_text(row, ANSWER_KEYS)
    if prompt and answer:
        return prompt, answer
    return None


def add_record(records, prompt: str, response: str, category: str, source: str):
    prompt = clean(prompt)
    response = clean(response)
    if not prompt or len(response) < 2:
        return
    records.append({
        "prompt": "user: " + prompt + "\nassistant: ",
        "response": response,
        "category": category,
        "source": source,
    })


def load_code_dataset(dataset_id: str, requested: int, seed: int, source_label: str):
    from datasets import load_dataset

    stream = load_dataset(dataset_id, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=min(10_000, max(100, requested * 2)))
    records = []
    scanned = 0
    for row in stream:
        scanned += 1
        pair = pair_from_row(row)
        if pair:
            prompt, response = pair
            if len(prompt) >= 6 and len(response) >= 8:
                add_record(records, "solve this programming task. provide a direct answer:\n" + prompt,
                           response, "coding", source_label)
        if len(records) >= requested:
            break
    print(f"{source_label}: accepted {len(records):,}/{requested:,} requested coding examples (scanned {scanned:,} rows)")
    return records


def load_story_dataset(requested: int, seed: int):
    from datasets import load_dataset

    stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    stream = stream.shuffle(seed=seed + 1, buffer_size=min(10_000, max(100, requested * 2)))
    records = []
    scanned = 0
    for row in stream:
        scanned += 1
        story = clean(row.get("text"))
        if len(story) < 70:
            continue

        # Start with a short excerpt, then ask for a continuation in the same style.
        cut = min(100, max(45, len(story) // 3))
        sentence_end = next((i for i in range(cut, min(len(story), cut + 45))
                             if story[i] in ".!?\n"), None)
        if sentence_end is not None:
            cut = sentence_end + 1
        prefix = story[:cut].strip()
        continuation = story[cut:].strip()[:220]
        if len(prefix) < 25 or len(continuation) < 15:
            continue
        add_record(records, "continue this story in the same style:\n" + prefix, continuation,
                   "story", "roneneldan/TinyStories")
        if len(records) >= requested:
            break
    print(f"TinyStories: accepted {len(records):,}/{requested:,} requested story examples (scanned {scanned:,} rows)")
    return records


def make_math_examples(count: int, seed: int):
    rng = random.Random(seed + 2)
    items = ["apples", "books", "stickers", "blocks", "stars", "coins", "shells", "crayons"]
    records = []
    for index in range(count):
        operation = index % 3
        if operation == 0:
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            result = a + b
            if index % 2:
                thing = rng.choice(items)
                prompt = f"You have {a} {thing} and get {b} more. How many {thing} do you have now?"
                response = f"You have {a} + {b} = {result} {thing}."
            else:
                prompt = f"What is {a} + {b}? Give the numeric answer."
                response = f"{a} + {b} = {result}."
        elif operation == 1:
            a = rng.randint(0, 50)
            b = rng.randint(0, a)
            result = a - b
            if index % 2:
                thing = rng.choice(items)
                prompt = f"You have {a} {thing} and give away {b}. How many remain?"
                response = f"{a} - {b} = {result}, so {result} {thing} remain."
            else:
                prompt = f"Calculate {a} - {b}. Give the numeric answer."
                response = f"{a} - {b} = {result}."
        else:
            a, b = rng.randint(0, 12), rng.randint(0, 12)
            result = a * b
            if index % 2:
                thing = rng.choice(items)
                prompt = f"There are {a} bags with {b} {thing} in each bag. How many {thing} altogether?"
                response = f"{a} * {b} = {result}, so there are {result} {thing} altogether."
            else:
                prompt = f"What is {a} * {b}? Give the numeric answer."
                response = f"{a} * {b} = {result}."
        add_record(records, prompt, response, "math", "synthetic-arithmetic")
    print(f"Synthetic arithmetic: generated {len(records):,} examples")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coding-samples", type=int, default=2000)
    parser.add_argument("--story-samples", type=int, default=1500)
    parser.add_argument("--math-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    for name in ("coding_samples", "story_samples", "math_samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Run: python -m pip install -r requirements-data.txt") from exc

    code_feedback = load_code_dataset("m-a-p/CodeFeedback-Filtered-Instruction",
                                      args.coding_samples // 2, args.seed, "m-a-p/CodeFeedback-Filtered-Instruction")
    code_alpaca = load_code_dataset("graycatHCO3/CodeAlpaca-20K-Python",
                                    args.coding_samples - args.coding_samples // 2, args.seed + 17,
                                    "graycatHCO3/CodeAlpaca-20K-Python")
    stories = load_story_dataset(args.story_samples, args.seed)
    math_rows = make_math_examples(args.math_samples, args.seed)

    rows = code_feedback + code_alpaca + stories + math_rows
    unique = {}
    for row in rows:
        key = (row["prompt"], row["response"])
        unique.setdefault(key, row)
    rows = list(unique.values())
    if len(rows) < 100:
        raise SystemExit(f"Only {len(rows)} usable hybrid examples were built; cannot train safely.")

    random.Random(args.seed).shuffle(rows)
    validation_count = max(1, int(len(rows) * args.validation_fraction))
    validation = rows[:validation_count]
    training = rows[validation_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    valid_path = args.output_dir / "validation.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training), encoding="utf-8")
    valid_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in validation), encoding="utf-8")
    manifest = {
        "format": "prompt/response JSONL for response-masked supervised fine-tuning",
        "seed": args.seed,
        "requested_samples": {
            "coding_total": args.coding_samples,
            "coding_codefeedback": args.coding_samples // 2,
            "coding_codealpaca": args.coding_samples - args.coding_samples // 2,
            "stories": args.story_samples,
            "synthetic_arithmetic": args.math_samples,
        },
        "accepted_total": len(rows),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "categories_total": dict(Counter(row["category"] for row in rows)),
        "categories_training": dict(Counter(row["category"] for row in training)),
        "categories_validation": dict(Counter(row["category"] for row in validation)),
        "sources": {
            "coding": [
                {"id": "m-a-p/CodeFeedback-Filtered-Instruction", "note": "Review dataset card and source terms before reuse."},
                {"id": "graycatHCO3/CodeAlpaca-20K-Python", "note": "Review dataset card and source terms before reuse."},
            ],
            "stories": [{"id": "roneneldan/TinyStories", "note": "Review current dataset card and license."}],
            "math": [{"id": "synthetic-arithmetic", "note": "Generated locally by this script."}],
        },
        "note": "Streaming dataset samples are requested counts, not guaranteed counts; review each current source card and terms.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"train file: {train_path}")
    print(f"validation file: {valid_path}")


if __name__ == "__main__":
    main()
