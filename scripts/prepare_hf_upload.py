"""Upload the unified training dataset to HuggingFace Hub."""

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi

DATA_FILE = "data/tibeb_unified_train.jsonl"
DEFAULT_REPO = "nahommohan/tibeb-training-data"


def get_stats(filepath):
    sources = {}
    total = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            total += 1
    return sources, total


def generate_readme(sources, total, size_mb):
    rows = "\n".join(
        f"| {src} | {count:,} | {count/total*100:.1f}% |"
        for src, count in sorted(sources.items(), key=lambda x: -x[1])
    )
    return f"""---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: source
    dtype: string
  splits:
  - name: train
    num_examples: {total}
license: cc-by-nc-4.0
language:
- am
- en
tags:
- amharic
- finance
- ethiopia
- instruction-tuning
---

# Tibeb Training Data

Unified training dataset for **Tibeb AI** — Ethiopia's Amharic financial assistant.

## Stats

- **Total rows**: {total:,}
- **File size**: {size_mb:.1f} MB
- **Languages**: Amharic, English, Mixed

## Sources

| Source | Rows | % |
|--------|------|---|
{rows}
| **TOTAL** | **{total:,}** | **100%** |

## Schema

Each row is a JSON object with:
- `instruction` — the task or prompt
- `input` — optional additional context
- `output` — the expected response
- `source` — dataset origin

Some rows include extra fields like `task_type`, `topic`, or `address_form`.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{DEFAULT_REPO}", split="train")
```
"""


def main():
    parser = argparse.ArgumentParser(description="Upload Tibeb training data to HuggingFace")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})")
    args = parser.parse_args()
    repo_id = args.repo

    filepath = Path(DATA_FILE)
    if not filepath.exists():
        print(f"Error: {DATA_FILE} not found. Run merge_datasets.py first.")
        return

    size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"File: {DATA_FILE} ({size_mb:.1f} MB)")

    print("Counting rows and sources...")
    sources, total = get_stats(filepath)
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src:30s} {count:>7,} ({count/total*100:.1f}%)")
    print(f"  {'TOTAL':30s} {total:>7,}")

    api = HfApi()

    print(f"\nUploading {DATA_FILE} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(filepath),
        path_in_repo="tibeb_unified_train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  Dataset uploaded.")

    print("Updating README.md...")
    readme = generate_readme(sources, total, size_mb)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  README updated.")

    print(f"\nDone! View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
