"""Download small, configurable text samples from public Hugging Face datasets.

Examples:
  python -m pip install -r requirements-data.txt
  python scripts/prepare_datasets.py --dataset tinystories --split train --max-samples 50000
  python scripts/prepare_datasets.py --dataset fineweb-edu --split train --max-samples 10000
  python scripts/prepare_datasets.py --dataset smol-smoltalk --split train --max-samples 20000

This script writes plain UTF-8 text files for the existing next-token trainer.
It does not download model weights or upload any data. Check each dataset card
and its current terms before training or redistributing derived data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

DATASETS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "text_columns": ("text",),
        "license_note": "CDLA-Sharing-1.0; review terms before reuse",
    },
    "fineweb-edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "text_columns": ("text",),
        "license_note": "ODC-By 1.0 and Common Crawl terms; review current terms",
    },
    "smol-smoltalk": {
        "path": "HuggingFaceTB/smol-smoltalk",
        "text_columns": ("messages", "text", "conversation"),
        "license_note": "Apache-2.0 dataset card; verify source and current terms",
    },
}


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def row_to_text(row: dict[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        value = row.get(column)
        if isinstance(value, str):
            return clean_text(value)
        if isinstance(value, list):
            # Chat datasets commonly store messages as {role, content} records.
            parts = []
            for message in value:
                if isinstance(message, dict):
                    role = str(message.get("role", "assistant")).strip()
                    content = message.get("content", "")
                    if isinstance(content, str) and content.strip():
                        parts.append(f"{role}: {clean_text(content)}")
            if parts:
                return "\n".join(parts)
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                if isinstance(item, str) and item.strip():
                    parts.append(f"{key}: {clean_text(item)}")
            if parts:
                return "\n".join(parts)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if not 0 <= args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if args.min_chars < 1:
        parser.error("--min-chars must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency. Run: pip install -r requirements-data.txt"
        ) from exc

    spec = DATASETS[args.dataset]
    print(f"dataset: {spec['path']} ({args.split})")
    print(f"license note: {spec['license_note']}")
    print(f"maximum accepted samples: {args.max_samples:,}")
    print("loading in streaming mode; this does not download the entire dataset")

    stream = load_dataset(spec["path"], split=args.split, streaming=True)
    # Deterministic shuffle buffer avoids simply taking the first shard rows.
    stream = stream.shuffle(seed=args.seed, buffer_size=min(10_000, max(100, args.max_samples)))
    texts: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    skipped = 0

    for row in stream:
        text = row_to_text(row, spec["text_columns"])
        if len(text) < args.min_chars:
            skipped += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        texts.append(text)
        if len(texts) >= args.max_samples:
            break

    if not texts:
        raise SystemExit("No usable text found. Check the dataset card and its schema.")

    # Split after deduplication, then write one document per line to preserve the
    # trainer's simple text-file interface while retaining paragraph boundaries.
    import random
    random.Random(args.seed).shuffle(texts)
    validation_count = max(1, int(len(texts) * args.validation_fraction)) if args.validation_fraction else 0
    validation = texts[:validation_count]
    training = texts[validation_count:]

    train_dir = args.output_dir / "train"
    valid_dir = args.output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_dir / f"{args.dataset.replace('-', '_')}.txt"
    valid_path = valid_dir / f"{args.dataset.replace('-', '_')}.txt"
    train_path.write_text("\n\n".join(training) + "\n", encoding="utf-8")
    if validation:
        valid_path.write_text("\n\n".join(validation) + "\n", encoding="utf-8")

    report = {
        "dataset": spec["path"],
        "split": args.split,
        "license_note": spec["license_note"],
        "seed": args.seed,
        "accepted_unique_samples": len(texts),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "duplicates_skipped": duplicates,
        "short_or_empty_rows_skipped": skipped,
        "training_file": str(train_path),
        "validation_file": str(valid_path) if validation else None,
        "note": "Record dataset revision and re-check source terms before redistribution.",
    }
    report_path = args.output_dir / f"{args.dataset.replace('-', '_')}_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
