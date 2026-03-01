"""
Merge all Amharic training data into one unified dataset for Aya fine-tuning.

Sources:
1. amharic_instructions_train.jsonl  — 122K general Amharic instructions
2. amharic_mt_train.jsonl            — 200K Amharic-English translation pairs
3. synthetic_qa/raw_generated.json   — 60 Tibeb financial conversations

Output format (Alpaca-style, works with LLaMA Factory and TRL):
{"instruction": "...", "input": "...", "output": "..."}
"""
import json
from pathlib import Path

def load_jsonl(filepath):
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def convert_financial_conversation(pair):
    """
    Convert a Tibeb financial conversation into instruction format.
    Each exchange becomes a separate training example.
    """
    examples = []
    conv = pair.get("conversation", [])
    profile = pair.get("profile", {})
    topic = pair.get("topic", "")

    for i in range(0, len(conv) - 1, 2):
        if i + 1 < len(conv):
            user_msg = conv[i]["content"]
            assistant_msg = conv[i + 1]["content"]

            examples.append({
                "instruction": user_msg,
                "input": "",
                "output": assistant_msg,
                "source": "tibeb_financial",
                "topic": topic,
                "address_form": profile.get("address_form", ""),
            })
    return examples

def main():
    output_file = "data/tibeb_unified_train.jsonl"
    stats = {}
    total = 0

    with open(output_file, "w", encoding="utf-8") as out:

        # Source 1 — General Amharic instructions
        print("Loading Amharic instructions...")
        rows = load_jsonl("data/amharic_instructions_train.jsonl")
        count = 0
        for row in rows:
            if row.get("instruction") and row.get("output"):
                out.write(json.dumps({
                    "instruction": row["instruction"],
                    "input": row.get("input", ""),
                    "output": row["output"],
                    "source": "ethionlp_instructions",
                }, ensure_ascii=False) + "\n")
                count += 1
        stats["amharic_instructions"] = count
        total += count
        print(f"  Added {count} instruction examples")

        # Source 2 — Machine translation pairs
        print("Loading MT pairs...")
        rows = load_jsonl("data/amharic_mt_train.jsonl")
        count = 0
        for row in rows:
            if row.get("instruction") and row.get("output"):
                out.write(json.dumps({
                    "instruction": row["instruction"],
                    "input": row.get("input", ""),
                    "output": row["output"],
                    "source": "ethionlp_mt",
                }, ensure_ascii=False) + "\n")
                count += 1
        stats["mt_pairs"] = count
        total += count
        print(f"  Added {count} MT pairs")

        # Source 3 — Tibeb financial conversations (highest priority)
        print("Loading Tibeb financial conversations...")
        with open("data/synthetic_qa/raw_generated.json", encoding="utf-8") as f:
            pairs = json.load(f)
        count = 0
        for pair in pairs:
            examples = convert_financial_conversation(pair)
            for ex in examples:
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
        stats["tibeb_financial"] = count
        total += count
        print(f"  Added {count} financial examples")

    print(f"\n{'='*50}")
    print(f"Unified dataset saved: {output_file}")
    print(f"\nBreakdown:")
    for source, count in stats.items():
        pct = count / total * 100
        print(f"  {source:30s} {count:6d} ({pct:.1f}%)")
    print(f"  {'TOTAL':30s} {total:6d}")

    # Check file size
    size_mb = Path(output_file).stat().st_size / 1024 / 1024
    print(f"\nFile size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
