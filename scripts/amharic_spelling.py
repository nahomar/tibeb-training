"""
Tibeb Amharic Spelling Normalizer
Enforces authoritative spellings across all training data and model output.

Covers:
  - ሰ/ሠ alternations (sadis)
  - ሀ/ሃ/ሐ/ኀ alternations (ha)
  - ጸ/ፀ alternations (tsa)
  - አ/ዐ alternations (a/ayin)
  - Common misspellings of financial, legal, and everyday terms
  - Punctuation normalization
"""

import json
import re

# -----------------------------------------------------------------------
# 1. ሰ/ሠ alternations — modern standard uses ሰ
# -----------------------------------------------------------------------
SADIS_CORRECTIONS = {
    "ሠላም":     "ሰላም",
    "ሠው":      "ሰው",
    "ሠራ":      "ሰራ",
    "ሠራተኛ":   "ሰራተኛ",
    "ሠራተኞች":  "ሰራተኞች",
    "ሠርቶ":     "ሰርቶ",
    "ሠርተው":   "ሰርተው",
    "ሠነድ":     "ሰነድ",
    "ሠንጠረዥ":  "ሰንጠረዥ",
    "ሠዓት":     "ሰዓት",
    "ሠማይ":     "ሰማይ",
    "ሠፈር":     "ሰፈር",
    "ሠፊ":      "ሰፊ",
    "ሠላሳ":     "ሰላሳ",
    "ሠብሰብ":   "ሰብሰብ",
    "ሠልፍ":     "ሰልፍ",
    "ሠሚ":      "ሰሚ",
    "ሠዋሰው":   "ሰዋሰው",
    "ሠይጣን":   "ሰይጣን",
    "ሠርገኛ":   "ሰርገኛ",
    "ሠርግ":     "ሰርግ",
    "ሠሪ":      "ሰሪ",
    "ሠላምታ":   "ሰላምታ",
    "ሠከነ":     "ሰከነ",
    "ሠከን":     "ሰከን",
    "ሠጠ":      "ሰጠ",
    "ሠጥቶ":     "ሰጥቶ",
    "ሠጪ":      "ሰጪ",
    "ሠበሰበ":   "ሰበሰበ",
    "ሠዎች":     "ሰዎች",
}

# -----------------------------------------------------------------------
# 2. ሀ/ሃ/ሐ/ኀ alternations — standardize to ሀ
# -----------------------------------------------------------------------
HA_CORRECTIONS = {
    # ሃ → ሀ
    "ሃገር":     "ሀገር",
    "ሃብት":     "ሀብት",
    "ሃሳብ":     "ሀሳብ",
    "ሃኪም":     "ሀኪም",
    "ሃይል":     "ሀይል",
    "ሃይማኖት":  "ሀይማኖት",
    "ሃዘን":     "ሀዘን",
    "ሃምሳ":     "ሀምሳ",
    "ሃያ":      "ሀያ",
    "ሃብታም":   "ሀብታም",
    "ሃላፊ":     "ሀላፊ",
    "ሃላፊነት":  "ሀላፊነት",
    "ሃሰት":     "ሀሰት",
    "ሃገራዊ":   "ሀገራዊ",
    "ሃገሪቱ":   "ሀገሪቱ",
    "ሃገሮች":   "ሀገሮች",
    "ሃምሌ":     "ሀምሌ",
    "ሃገራችን":  "ሀገራችን",
    # ሐ → ሀ
    "ሐገር":     "ሀገር",
    "ሐብት":     "ሀብት",
    "ሐሳብ":     "ሀሳብ",
    "ሐኪም":     "ሀኪም",
    "ሐይል":     "ሀይል",
    "ሐይማኖት":  "ሀይማኖት",
    "ሐዘን":     "ሀዘን",
    "ሐምሳ":     "ሀምሳ",
    "ሐያ":      "ሀያ",
    "ሐምሌ":     "ሀምሌ",
    "ሐረር":     "ሀረር",
    "ሐረሪ":     "ሀረሪ",
    # ኀ → ሀ (archaic form)
    "ኀገር":     "ሀገር",
    "ኀብት":     "ሀብት",
    "ኀሳብ":     "ሀሳብ",
    "ኀይል":     "ሀይል",
    "ኀይማኖት":  "ሀይማኖት",
    "ኀዘን":     "ሀዘን",
}

# -----------------------------------------------------------------------
# 3. ጸ/ፀ alternations — standardize to ጸ
# -----------------------------------------------------------------------
TSA_CORRECTIONS = {
    "ፀሐይ":     "ጸሐይ",
    "ፀሎት":     "ጸሎት",
    "ፀጥታ":     "ጸጥታ",
    "ፀሐፊ":     "ጸሐፊ",
    "ፀሐፍት":    "ጸሐፍት",
    "ፀሀፊ":     "ጸሐፊ",
    "ፀሀይ":     "ጸሐይ",
    "ፀጋ":      "ጸጋ",
    "ፀረ":      "ጸረ",
    "ፀረ-":     "ጸረ-",
    "ፀጉር":     "ጸጉር",
    "ፀጉሪ":     "ጸጉሪ",
    "ፅሁፍ":     "ጽሑፍ",
    "ፅሑፍ":     "ጽሑፍ",
    "ፅሕፈት":    "ጽሕፈት",
    "ፅኑ":      "ጽኑ",
    "ፅዳት":     "ጽዳት",
    "ፅንሰ":     "ጽንሰ",
    "ፅዋ":      "ጽዋ",
}

# -----------------------------------------------------------------------
# 4. አ/ዐ alternations — standardize to simpler form
#    Note: some words traditionally use ዐ but modern usage prefers አ
# -----------------------------------------------------------------------
AYIN_CORRECTIONS = {
    "ዓስተዳደር":  "አስተዳደር",
    "ዐስተዳደር":  "አስተዳደር",
}

# -----------------------------------------------------------------------
# 5. Financial terms
# -----------------------------------------------------------------------
FINANCIAL_CORRECTIONS = {
    "አክስዮን":   "አክሲዮን",
    "ቲቢል":     "ቲ-ቢል",
    "ብርር":     "ብር",
    "ምንዛሪ":    "ምንዛሬ",
    "ምንዛሬዉ":  "ምንዛሬው",
    "ኢንቬስት":  "ኢንቨስት",
    "ኢንቬስትመንት": "ኢንቨስትመንት",
    "ቁጥባ":     "ቁጠባ",
    "ገበያዉ":   "ገበያው",
    "ፋይናንሰ":  "ፋይናንስ",
    "ኢኮኖሚይ":  "ኢኮኖሚ",
    "ኢንሹራንሰ":  "ኢንሹራንስ",
    "ዲቪዴንድ":  "ዲቪደንድ",
    "ዲቪዲንድ":  "ዲቪደንድ",
    "ታክሰ":     "ታክስ",
    "ጨረታዉ":   "ጨረታው",
}

# -----------------------------------------------------------------------
# 6. Address forms and common words
# -----------------------------------------------------------------------
ADDRESS_CORRECTIONS = {
    "እርሶ":     "እርስዎ",
    "እርሶን":    "እርስዎን",
    "እርሶው":    "እርስዎ",
    "ወይዘሮዋ":  "ወይዘሮ",
    "ወይዘሪቷ":  "ወይዘሪት",
}

# -----------------------------------------------------------------------
# 7. Common everyday words with known misspellings
# -----------------------------------------------------------------------
COMMON_CORRECTIONS = {
    "ኢትዮጲያ":  "ኢትዮጵያ",
    "ኢትዮጲያዊ": "ኢትዮጵያዊ",
    "ኢትዮጲያን":  "ኢትዮጵያን",
    "ኢትዮጲያዊነት": "ኢትዮጵያዊነት",
    "መንግሰት":  "መንግሥት",
    "ምርጫዉ":   "ምርጫው",
    "ፓርላማዉ":  "ፓርላማው",
    "ህገ-መንግስት": "ሕገ-መንግሥት",
    "ዩኒቨርሲቲዉ": "ዩኒቨርሲቲው",
    "ዩኒቬርሲቲ":  "ዩኒቨርሲቲ",
    "ቴክኖሎጂዉ": "ቴክኖሎጂው",
    "ኮምፒዩተር": "ኮምፒውተር",
    "ኮምፒዉተር": "ኮምፒውተር",
    "ኢንተሪኔት": "ኢንተርኔት",
    "ድረገጽ":   "ድረ-ገጽ",
    "ድረ ገጽ":  "ድረ-ገጽ",
    # Calendar
    "ህዳር":     "ኅዳር",
    "ታህሳስ":   "ታኅሣሥ",
}

# -----------------------------------------------------------------------
# 8. Institutions and proper nouns
# -----------------------------------------------------------------------
INSTITUTION_CORRECTIONS = {
    "ኢትዮ-ቴሌኮም":  "ኢትዮ ቴሌኮም",
    "ኢትዮቴሌኮም":   "ኢትዮ ቴሌኮም",
    "ቴሌቢር":       "ቴሌብር",
    "ብሄራዊ":       "ብሔራዊ",
    "ብሄር":        "ብሔር",
}


# -----------------------------------------------------------------------
# Build the combined dictionary
# -----------------------------------------------------------------------
WORD_CORRECTIONS = {}
WORD_CORRECTIONS.update(SADIS_CORRECTIONS)
WORD_CORRECTIONS.update(HA_CORRECTIONS)
WORD_CORRECTIONS.update(TSA_CORRECTIONS)
WORD_CORRECTIONS.update(AYIN_CORRECTIONS)
WORD_CORRECTIONS.update(FINANCIAL_CORRECTIONS)
WORD_CORRECTIONS.update(ADDRESS_CORRECTIONS)
WORD_CORRECTIONS.update(COMMON_CORRECTIONS)
WORD_CORRECTIONS.update(INSTITUTION_CORRECTIONS)


def normalize(text: str) -> str:
    """
    Correct known spelling variants to Tibeb authoritative forms.
    Uses substring replacement — safe for Amharic because the
    misspelled forms don't appear as substrings of other valid words.
    """
    for wrong, correct in WORD_CORRECTIONS.items():
        text = text.replace(wrong, correct)

    # Normalize common punctuation issues
    # Double Ethiopic periods → single
    text = text.replace("።።", "።")
    # Space before Ethiopic period
    text = re.sub(r"\s+።", "።", text)
    # Missing space after Ethiopic comma
    text = re.sub(r"፣(\S)", r"፣ \1", text)

    return text


def normalize_row(row: dict) -> dict:
    """Apply spelling normalization to instruction/input/output fields."""
    for field in ("instruction", "input", "output"):
        if row.get(field):
            row[field] = normalize(row[field])
    return row


def check_training_data(filepath: str) -> dict:
    """
    Scan a training data file for spelling issues.
    Supports both JSON arrays (.json) and JSONL (.jsonl) formats.
    Returns a report of what was found and fixed.
    """
    is_jsonl = filepath.endswith(".jsonl")

    if is_jsonl:
        data = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

    issues_found = []
    fixed_data = []

    for i, item in enumerate(data):
        item_str = json.dumps(item, ensure_ascii=False)
        fixed_str = normalize(item_str)

        if item_str != fixed_str:
            for wrong, correct in WORD_CORRECTIONS.items():
                if wrong in item_str:
                    issues_found.append({
                        "index": i,
                        "wrong": wrong,
                        "correct": correct,
                        "source": item.get("source", item.get("topic", "")),
                    })

        fixed_data.append(json.loads(fixed_str))

    ext = ".jsonl" if is_jsonl else ".json"
    corrected_path = filepath.replace(ext, f"_corrected{ext}")

    with open(corrected_path, "w", encoding="utf-8") as f:
        if is_jsonl:
            for row in fixed_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    return {
        "total_items": len(data),
        "items_with_issues": len(set(i["index"] for i in issues_found)),
        "total_corrections": len(issues_found),
        "issues": issues_found,
        "corrected_file": corrected_path,
    }


def stats():
    """Print dictionary statistics."""
    print(f"Spelling dictionary statistics:")
    print(f"  ሰ/ሠ corrections:     {len(SADIS_CORRECTIONS)}")
    print(f"  ሀ/ሃ/ሐ/ኀ corrections: {len(HA_CORRECTIONS)}")
    print(f"  ጸ/ፀ corrections:     {len(TSA_CORRECTIONS)}")
    print(f"  አ/ዐ corrections:     {len(AYIN_CORRECTIONS)}")
    print(f"  Financial terms:      {len(FINANCIAL_CORRECTIONS)}")
    print(f"  Address forms:        {len(ADDRESS_CORRECTIONS)}")
    print(f"  Common words:         {len(COMMON_CORRECTIONS)}")
    print(f"  Institutions:         {len(INSTITUTION_CORRECTIONS)}")
    print(f"  ─────────────────────────")
    print(f"  TOTAL corrections:    {len(WORD_CORRECTIONS)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        stats()
        sys.exit(0)

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_qa/raw_generated.json"
    print(f"Checking: {filepath}")

    stats()
    print()

    report = check_training_data(filepath)
    print(f"Results:")
    print(f"  Total items:         {report['total_items']}")
    print(f"  Items with issues:   {report['items_with_issues']}")
    print(f"  Total corrections:   {report['total_corrections']}")
    print(f"  Corrected file:      {report['corrected_file']}")

    if report["issues"]:
        print(f"\nIssues found:")
        for issue in report["issues"][:30]:
            print(f"  [{issue['index']}] {issue['source']} — {issue['wrong']} -> {issue['correct']}")
        if len(report["issues"]) > 30:
            print(f"  ... and {len(report['issues']) - 30} more")
    else:
        print("\n  No spelling issues found.")
