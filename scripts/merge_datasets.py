"""
Merge all training data sources into a single unified JSONL dataset.

Pipeline: load sources -> normalize spelling -> deduplicate -> shuffle -> write.

Usage:
    python scripts/merge_datasets.py
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path

# Allow import from scripts/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amharic_spelling import normalize, normalize_row

SENTIMENT_INSTRUCTION = "የሚከተለው ጽሑፍ ምን ዓይነት ስሜት ያሳያል?"
NEWS_INSTRUCTION = "የዚህን ጽሑፍ ምድብ ለይ።"
ALFFA_INSTRUCTION = "ይህን የአማርኛ ንግግር ጽሑፍ ተመልከት።"


def load_jsonl(filepath):
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def content_hash(row):
    """Hash instruction+input+output for deduplication."""
    key = f"{row.get('instruction', '')}|{row.get('input', '')}|{row.get('output', '')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def truncate_at_boundary(text, max_chars=500):
    """Truncate text at a sentence or word boundary instead of mid-word."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_stop = truncated.rfind("።")
    if last_stop > max_chars * 0.5:
        return truncated[:last_stop + 1]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space]
    return truncated


def convert_financial_conversation(pair):
    """Extract instruction/output pairs from a multi-turn conversation."""
    examples = []
    conv = pair.get("conversation", [])
    profile = pair.get("profile", {})
    topic = pair.get("topic", "")
    for i in range(0, len(conv) - 1, 2):
        if i + 1 < len(conv):
            examples.append({
                "instruction": conv[i]["content"],
                "input": "",
                "output": conv[i + 1]["content"],
                "source": "tibeb_financial",
                "topic": topic,
                "address_form": profile.get("address_form", ""),
            })
    return examples


def main():
    output_file = "data/tibeb_unified_train.jsonl"
    all_rows = []
    stats = {}

    # ---------------------------------------------------------------
    # 1. Amharic instructions
    # ---------------------------------------------------------------
    print("Loading Amharic instructions...")
    rows = load_jsonl("data/amharic_instructions_train.jsonl")
    count = 0
    for row in rows:
        if row.get("instruction") and row.get("output"):
            all_rows.append({
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row["output"],
                "source": "ethionlp_instructions",
            })
            count += 1
    stats["amharic_instructions"] = count
    print(f"  Added {count}")

    # ---------------------------------------------------------------
    # 2. MT pairs (Only keep Amharic output)
    # ---------------------------------------------------------------
    print("Loading MT pairs (filtering for Amharic output)...")
    rows = load_jsonl("data/amharic_mt_train.jsonl")
    count = 0
    from fetch_amharic_corpus import has_amharic
    
    for row in rows:
        if row.get("instruction") and row.get("output"):
            # Only keep EN -> AM translation (output must be Amharic)
            if has_amharic(row["output"], min_ratio=0.3):
                all_rows.append({
                    "instruction": row["instruction"],
                    "input": row.get("input", ""),
                    "output": row["output"],
                    "source": "ethionlp_mt",
                })
                count += 1
    stats["mt_pairs"] = count
    print(f"  Added {count} (filtered from {len(rows)})")

    # ---------------------------------------------------------------
    # 3. Sentiment data (labels translated to Amharic)
    # ---------------------------------------------------------------
    SENTIMENT_LABELS = {"positive": "አዎንታዊ", "negative": "አሉታዊ", "neutral": "ገለልተኛ"}
    if Path("data/ethiosenti_train.jsonl").exists():
        print("Loading EthioSenti...")
        rows = load_jsonl("data/ethiosenti_train.jsonl")
        count = 0
        for row in rows:
            if row.get("tweet"):
                label = SENTIMENT_LABELS.get(row.get("label", ""), row.get("label", ""))
                all_rows.append({
                    "instruction": SENTIMENT_INSTRUCTION,
                    "input": row["tweet"],
                    "output": label,
                    "source": "ethiosenti",
                })
                count += 1
        stats["ethiosenti"] = count
        print(f"  Added {count}")

    # ---------------------------------------------------------------
    # 4. News classification (categories already in Amharic)
    # ---------------------------------------------------------------
    if Path("data/amharic_news_class.jsonl").exists():
        print("Loading news classification...")
        rows = load_jsonl("data/amharic_news_class.jsonl")
        count = 0
        for row in rows:
            if row.get("article"):
                all_rows.append({
                    "instruction": NEWS_INSTRUCTION,
                    "input": truncate_at_boundary(row["article"], max_chars=500),
                    "output": row.get("category", ""),
                    "source": "amharic_news",
                })
                count += 1
        stats["news_classification"] = count
        print(f"  Added {count}")

    # ---------------------------------------------------------------
    # 5. Amharic sentiments corpus
    # ---------------------------------------------------------------
    if Path("data/amharic_sentiments.jsonl").exists():
        print("Loading Amharic sentiments corpus...")
        rows = load_jsonl("data/amharic_sentiments.jsonl")
        count = 0
        for row in rows:
            if row.get("Amharic"):
                all_rows.append({
                    "instruction": SENTIMENT_INSTRUCTION,
                    "input": row["Amharic"],
                    "output": row.get("sentiment", ""),
                    "source": "amharic_sentiments",
                })
                count += 1
        stats["amharic_sentiments"] = count
        print(f"  Added {count}")

    # ---------------------------------------------------------------
    # 6. ALFFA voice transcriptions
    # NOTE: ALFFA data has ASR-style morpheme spacing (e.g. "የ ትግራይ").
    # Converted to continuation format to avoid echo (input==output) problem.
    # ---------------------------------------------------------------
    if Path("data/alffa_transcriptions.json").exists():
        print("Loading ALFFA transcriptions...")
        with open("data/alffa_transcriptions.json", encoding="utf-8") as f:
            alffa = json.load(f)
        count = 0
        for item in alffa:
            text = item.get("text", "").strip()
            if not text or len(text) < 50:
                continue
            # Split into continuation pairs instead of echo
            sentences = [s for s in re.split(r"(?<=።)\s*", text) if s.strip()]
            if len(sentences) >= 2:
                split = len(sentences) // 2
                prefix = "። ".join(sentences[:split]) + "።"
                suffix = "። ".join(sentences[split:])
                if suffix and not suffix.endswith("።"):
                    suffix += "።"
                all_rows.append({
                    "instruction": ALFFA_INSTRUCTION,
                    "input": prefix,
                    "output": suffix,
                    "source": "alffa_voice",
                })
                count += 1
        stats["alffa_voice"] = count
        print(f"  Added {count}")

    # ---------------------------------------------------------------
    # 7. Aya Collection (Amharic subset)
    # ---------------------------------------------------------------
    if Path("data/aya_amharic_train.jsonl").exists():
        print("Loading Aya Collection (Amharic)...")
        rows = load_jsonl("data/aya_amharic_train.jsonl")
        count = 0
        for row in rows:
            if row.get("instruction") and row.get("output"):
                all_rows.append({
                    "instruction": row["instruction"],
                    "input": row.get("input", ""),
                    "output": row["output"],
                    "source": "aya_collection",
                })
                count += 1
        stats["aya_amharic"] = count
        print(f"  Added {count}")

    # ---------------------------------------------------------------
    # 8. Sujet Finance Instruct (English — for cross-lingual transfer)
    # Skipped because we want 100% Amharic output for quality training.
    # ---------------------------------------------------------------
    # if Path("data/sujet_finance_instruct.jsonl").exists():
    #     print("Loading Sujet Finance Instruct (English, cross-lingual)...")
    #     rows = load_jsonl("data/sujet_finance_instruct.jsonl")
    #     count = 0
    #     for row in rows:
    #         prompt = row.get("user_prompt") or row.get("inputs", "")
    #         answer = row.get("answer", "")
    #         system_prompt = row.get("system_prompt", "")
    #         if prompt and answer and system_prompt:
    #             all_rows.append({
    #                 "instruction": system_prompt,
    #                 "input": prompt,
    #                 "output": answer,
    #                 "source": "sujet_finance",
    #                 "task_type": row.get("task_type", ""),
    #             })
    #             count += 1
    #     stats["sujet_finance"] = count
    #     print(f"  Added {count}")

    # ---------------------------------------------------------------
    # 9. Native Amharic corpora (Wikipedia, CC-100, XL-Sum, etc.)
    # ---------------------------------------------------------------
    native_dir = Path("data/native_amharic")
    if native_dir.exists():
        for corpus_file in sorted(native_dir.glob("*.jsonl")):
            source_name = corpus_file.stem
            print(f"Loading native corpus: {source_name}...")
            rows = load_jsonl(str(corpus_file))
            count = 0
            for row in rows:
                if row.get("instruction") and row.get("output"):
                    all_rows.append({
                        "instruction": row["instruction"],
                        "input": row.get("input", ""),
                        "output": row["output"],
                        "source": row.get("source", source_name),
                    })
                    count += 1
            stats[source_name] = count
            print(f"  Added {count}")
    else:
        print("Skipping native Amharic corpora (not found)"
              " — run fetch_amharic_corpus.py first")

    # ---------------------------------------------------------------
    # 10. Tibeb financial conversations (synthetic)
    # Upsampled — these are highest-value data for Tibeb persona
    # ---------------------------------------------------------------
    for synth_file, source_name, upsample in [
        ("data/synthetic_qa/raw_generated.json", "tibeb_financial", 5),
        ("data/synthetic_qa/financial_conversations.json", "tibeb_financial_v2", 5),
        ("data/synthetic_qa/general_amharic.json", "tibeb_general", 3),
        ("data/synthetic_qa/general_conversations.json", "tibeb_general_v2", 3),
        ("data/synthetic_qa/voice_conversations.json", "tibeb_voice", 5),
        ("data/synthetic_qa/safety_conversations.json", "tibeb_safety", 5),
        ("data/synthetic_qa/deep_conversations.json", "tibeb_deep", 3),
    ]:
        if Path(synth_file).exists():
            print(f"Loading {source_name}...")
            with open(synth_file, encoding="utf-8") as f:
                pairs = json.load(f)
            count = 0
            for pair in pairs:
                for ex in convert_financial_conversation(pair):
                    ex["source"] = source_name
                    for _ in range(upsample):
                        all_rows.append(ex)
                        count += 1
            stats[source_name] = count
            print(f"  Added {count} ({upsample}x upsample from {len(pairs)} convos)")
        else:
            print(f"Skipping {source_name} ({synth_file} not found)"
                  " — run generate_synthetic_data.py")

    raw_total = len(all_rows)

    # ---------------------------------------------------------------
    # Post-processing: normalize, deduplicate, filter, shuffle
    # ---------------------------------------------------------------
    print(f"\nTotal raw rows: {raw_total}")

    print("Filtering rows requiring output to contain at least 30% Amharic characters...")
    from fetch_amharic_corpus import has_amharic
    filtered_rows = []
    for row in all_rows:
        if has_amharic(row.get("output", ""), min_ratio=0.3):
            filtered_rows.append(row)
    filtered_removed = len(all_rows) - len(filtered_rows)
    all_rows = filtered_rows
    print(f"  Removed {filtered_removed} rows without enough Amharic characters")

    print("Applying spelling normalization...")
    all_rows = [normalize_row(row) for row in all_rows]

    print("Deduplicating...")
    seen = set()
    unique_rows = []
    for row in all_rows:
        h = content_hash(row)
        if h not in seen:
            seen.add(h)
            unique_rows.append(row)
    dupes_removed = raw_total - len(unique_rows)
    all_rows = unique_rows
    print(f"  Removed {dupes_removed} duplicates")

    print("Shuffling (seed=42)...")
    random.seed(42)
    random.shuffle(all_rows)

    # ---------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------
    total = len(all_rows)
    print(f"\nWriting {total} rows to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as out:
        for row in all_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"Unified dataset: {output_file}")
    print(f"\nBreakdown (pre-dedup counts):")
    for source, count in stats.items():
        pct = count / raw_total * 100
        print(f"  {source:30s} {count:7d} ({pct:.1f}%)")
    print(f"  {'RAW TOTAL':30s} {raw_total:7d}")
    print(f"  {'Duplicates removed':30s} {dupes_removed:7d}")
    print(f"  {'FINAL TOTAL':30s} {total:7d}")
    size_mb = Path(output_file).stat().st_size / 1024 / 1024
    print(f"\nFile size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
