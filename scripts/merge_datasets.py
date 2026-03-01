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
    examples = []
    conv = pair.get("conversation", [])
    profile = pair.get("profile", {})
    topic = pair.get("topic", "")
    for i in range(0, len(conv) - 1, 2):
        if i + 1 < len(conv):
            examples.append({
                "instruction": conv[i]["content"],
                "input": "",
                "output": conv[i+1]["content"],
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

        # 1. Amharic instructions
        print("Loading Amharic instructions...")
        rows = load_jsonl("data/amharic_instructions_train.jsonl")
        count = 0
        for row in rows:
            if row.get("instruction") and row.get("output"):
                out.write(json.dumps({"instruction": row["instruction"], "input": row.get("input", ""), "output": row["output"], "source": "ethionlp_instructions"}, ensure_ascii=False) + "\n")
                count += 1
        stats["amharic_instructions"] = count
        total += count
        print(f"  Added {count}")

        # 2. MT pairs
        print("Loading MT pairs...")
        rows = load_jsonl("data/amharic_mt_train.jsonl")
        count = 0
        for row in rows:
            if row.get("instruction") and row.get("output"):
                out.write(json.dumps({"instruction": row["instruction"], "input": row.get("input", ""), "output": row["output"], "source": "ethionlp_mt"}, ensure_ascii=False) + "\n")
                count += 1
        stats["mt_pairs"] = count
        total += count
        print(f"  Added {count}")

        # 3. Sentiment data
        if Path("data/ethiosenti_train.jsonl").exists():
            print("Loading EthioSenti...")
            rows = load_jsonl("data/ethiosenti_train.jsonl")
            count = 0
            for row in rows:
                if row.get("tweet"):
                    out.write(json.dumps({"instruction": "የሚከተለው ጽሑፍ ምን አይነት ስሜት ያሳያል?", "input": row["tweet"], "output": row.get("label", ""), "source": "ethiosenti"}, ensure_ascii=False) + "\n")
                    count += 1
            stats["ethiosenti"] = count
            total += count
            print(f"  Added {count}")

        # 4. News classification
        if Path("data/amharic_news_class.jsonl").exists():
            print("Loading news classification...")
            rows = load_jsonl("data/amharic_news_class.jsonl")
            count = 0
            for row in rows:
                if row.get("article"):
                    out.write(json.dumps({"instruction": "የዚህን ጽሑፍ ምድብ ለይ።", "input": row["article"][:500], "output": row.get("category", ""), "source": "amharic_news"}, ensure_ascii=False) + "\n")
                    count += 1
            stats["news_classification"] = count
            total += count
            print(f"  Added {count}")

        # 5. Amharic sentiments corpus
        if Path("data/amharic_sentiments.jsonl").exists():
            print("Loading Amharic sentiments corpus...")
            rows = load_jsonl("data/amharic_sentiments.jsonl")
            count = 0
            for row in rows:
                if row.get("Amharic"):
                    out.write(json.dumps({"instruction": "የሚከተለው ጽሑፍ ምን ዓይነት ስሜት ያሳያል?", "input": row["Amharic"], "output": row.get("sentiment", ""), "source": "amharic_sentiments"}, ensure_ascii=False) + "\n")
                    count += 1
            stats["amharic_sentiments"] = count
            total += count
            print(f"  Added {count}")

        # 6. ALFFA voice transcriptions
        if Path("data/alffa_transcriptions.json").exists():
            print("Loading ALFFA transcriptions...")
            with open("data/alffa_transcriptions.json", encoding="utf-8") as f:
                alffa = json.load(f)
            count = 0
            for item in alffa:
                if item.get("text"):
                    out.write(json.dumps({"instruction": "ይህን አማርኛ ዓረፍተ ነገር ድገም።", "input": "", "output": item["text"], "source": "alffa_voice", "speaker_id": item.get("speaker_id", "")}, ensure_ascii=False) + "\n")
                    count += 1
            stats["alffa_voice"] = count
            total += count
            print(f"  Added {count}")

        # 7. Tibeb financial conversations
        print("Loading Tibeb financial conversations...")
        with open("data/synthetic_qa/raw_generated.json", encoding="utf-8") as f:
            pairs = json.load(f)
        count = 0
        for pair in pairs:
            for ex in convert_financial_conversation(pair):
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
        stats["tibeb_financial"] = count
        total += count
        print(f"  Added {count}")

    print(f"\n{'='*50}")
    print(f"Unified dataset: {output_file}")
    print(f"\nBreakdown:")
    for source, count in stats.items():
        pct = count / total * 100
        print(f"  {source:30s} {count:7d} ({pct:.1f}%)")
    print(f"  {'TOTAL':30s} {total:7d}")
    size_mb = Path(output_file).stat().st_size / 1024 / 1024
    print(f"\nFile size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
