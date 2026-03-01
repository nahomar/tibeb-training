"""
Tibeb Amharic Spelling Normalizer
Enforces authoritative spellings across all training data and model output.
"""

# Word-level corrections — only words we are 100% certain about
WORD_CORRECTIONS = {
    "ሠላም":    "ሰላም",
    "ሠው":     "ሰው",
    "ሃገር":    "ሀገር",
    "ሐገር":    "ሀገር",
    "እርሶ":    "እርስዎ",
    "አክስዮን":  "አክሲዮን",
    "ቲቢል":   "ቲ-ቢል",
    "ብርር":    "ብር",
    "ምንዛሪ":  "ምንዛሬ",
    "ኢትዮጲያ": "ኢትዮጵያ",
}


def normalize(text: str) -> str:
    """
    Correct known spelling variants to Tibeb authoritative forms.
    Only corrects words we are certain about.
    Leaves everything else untouched.
    """
    for wrong, correct in WORD_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text


def check_training_data(filepath: str) -> dict:
    """
    Scan a training data file for spelling issues.
    Returns a report of what was found and fixed.
    """
    import json

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    issues_found = []
    fixed_data = []

    for i, pair in enumerate(data):
        pair_str = json.dumps(pair, ensure_ascii=False)
        fixed_str = normalize(pair_str)

        if pair_str != fixed_str:
            # Find what changed
            for wrong, correct in WORD_CORRECTIONS.items():
                if wrong in pair_str:
                    issues_found.append({
                        "pair_index": i,
                        "wrong": wrong,
                        "correct": correct,
                        "topic": pair.get("topic", ""),
                        "name": pair.get("profile", {}).get("name", ""),
                    })

        fixed_data.append(json.loads(fixed_str))

    # Save corrected version
    corrected_path = filepath.replace(".json", "_corrected.json")
    with open(corrected_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    return {
        "total_pairs": len(data),
        "pairs_with_issues": len(set(i["pair_index"] for i in issues_found)),
        "total_corrections": len(issues_found),
        "issues": issues_found,
        "corrected_file": corrected_path,
    }


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_qa/raw_generated.json"
    print(f"Checking: {filepath}")
    report = check_training_data(filepath)
    print(f"\nResults:")
    print(f"  Total pairs:         {report['total_pairs']}")
    print(f"  Pairs with issues:   {report['pairs_with_issues']}")
    print(f"  Total corrections:   {report['total_corrections']}")
    print(f"  Corrected file:      {report['corrected_file']}")

    if report["issues"]:
        print(f"\nIssues found:")
        for issue in report["issues"][:10]:
            print(f"  [{issue['pair_index']}] {issue['name']} — {issue['wrong']} → {issue['correct']}")
    else:
        print("\n  No spelling issues found.")