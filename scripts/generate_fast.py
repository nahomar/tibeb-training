"""
Synthetic data generation respecting rate limits (8K output tokens/min).
Generates financial + general Amharic conversations for Tibeb.
"""

import anthropic
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_synthetic_data import (
    FINANCIAL_TOPICS, FINANCIAL_SCENARIOS, GENERAL_TOPICS,
    PROFILES, FINANCIAL_PROMPT, GENERAL_PROMPT,
)

# Rate limit: 8K output tokens/min → ~2 requests/min at 2000 tokens each
MAX_TOKENS = 2000
DELAY_BETWEEN_REQUESTS = 35  # seconds — safe for 8K/min limit


def generate_one(client, prompt_text):
    """Generate a single conversation with retry."""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = response.content[0].text.strip()

            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("["):
                        text = part
                        break

            messages = json.loads(text)
            if isinstance(messages, list) and len(messages) >= 4:
                return messages

        except json.JSONDecodeError:
            print(f"    JSON parse error (attempt {attempt+1})")
        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    Error (attempt {attempt+1}): {e}")
            time.sleep(5)

    return None


def run_financial(client, max_count=None):
    """Generate financial conversations."""
    output_file = "data/synthetic_qa/raw_generated.json"
    Path("data/synthetic_qa").mkdir(parents=True, exist_ok=True)

    existing = []
    if Path(output_file).exists():
        with open(output_file, encoding="utf-8") as f:
            existing = json.load(f)

    combos = []
    for topic in FINANCIAL_TOPICS:
        for profile in PROFILES:
            scenario = random.choice(FINANCIAL_SCENARIOS)
            combos.append((topic, profile, scenario))

    random.seed(42)
    random.shuffle(combos)

    if max_count:
        combos = combos[:max_count]
    combos = combos[len(existing):]

    total = len(existing) + len(combos)
    errors = 0
    start = time.time()

    print(f"\n=== Financial: {len(existing)} existing, {len(combos)} remaining, {total} target ===\n")

    for i, (topic, profile, scenario) in enumerate(combos):
        idx = len(existing) + 1
        print(f"  [{idx}/{total}] {profile['name']} — {topic[:50]}...")

        prompt_text = FINANCIAL_PROMPT.format(
            topic=topic, scenario=scenario,
            name=profile["name"], gender=profile["gender"],
            age=profile["age"], form=profile["form"],
        )

        messages = generate_one(client, prompt_text)

        if messages:
            existing.append({
                "topic": topic,
                "profile": {"name": profile["name"], "gender": profile["gender"],
                            "age": profile["age"], "address_form": profile["form"]},
                "conversation": messages,
                "type": "financial",
            })
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"    OK ({len(messages)} msgs)")
        else:
            errors += 1
            print(f"    FAILED")

        elapsed = time.time() - start
        done = i + 1
        rate = done / max(elapsed / 60, 0.1)
        eta = (len(combos) - done) / max(rate, 0.1)
        print(f"    [{rate:.1f}/min, ETA {eta:.0f}min]")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nFinancial done: {len(existing)} total, {errors} errors")
    return existing


def run_general(client, max_count=None):
    """Generate general Amharic conversations."""
    output_file = "data/synthetic_qa/general_amharic.json"
    Path("data/synthetic_qa").mkdir(parents=True, exist_ok=True)

    existing = []
    if Path(output_file).exists():
        with open(output_file, encoding="utf-8") as f:
            existing = json.load(f)

    combos = []
    for topic in GENERAL_TOPICS:
        for profile in PROFILES[:6]:
            combos.append((topic, profile))

    random.seed(43)
    random.shuffle(combos)

    if max_count:
        combos = combos[:max_count]
    combos = combos[len(existing):]

    total = len(existing) + len(combos)
    errors = 0

    print(f"\n=== General: {len(existing)} existing, {len(combos)} remaining, {total} target ===\n")

    for i, (topic, profile) in enumerate(combos):
        idx = len(existing) + 1
        print(f"  [{idx}/{total}] {profile['name']} — {topic[:50]}...")

        prompt_text = GENERAL_PROMPT.format(
            topic=topic, name=profile["name"],
            gender=profile["gender"], age=profile["age"],
        )

        messages = generate_one(client, prompt_text)

        if messages:
            existing.append({
                "topic": topic,
                "profile": {"name": profile["name"], "gender": profile["gender"],
                            "age": profile["age"]},
                "conversation": messages,
                "type": "general",
            })
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"    OK ({len(messages)} msgs)")
        else:
            errors += 1
            print(f"    FAILED")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nGeneral done: {len(existing)} total, {errors} errors")
    return existing


def main():
    client = anthropic.Anthropic()

    # Test API first
    print("Testing API connection...")
    try:
        r = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=20,
            messages=[{"role": "user", "content": "Say ሰላም"}],
        )
        print(f"  OK: {r.content[0].text.strip()}")
    except Exception as e:
        print(f"  API error: {e}")
        sys.exit(1)

    run_financial(client)
    run_general(client)

    print("\n=== All generation complete ===")


if __name__ == "__main__":
    main()
