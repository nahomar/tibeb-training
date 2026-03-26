"""Simple synchronous generator that actually works with rate limits."""
import anthropic
import json
import os
import random
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, "scripts")
from generate_50k import (
    FINANCIAL_TOPICS, GENERAL_TOPICS, VOICE_USSD_TOPICS,
    SAFETY_TOPICS, MULTI_TURN_TOPICS, PROFILES, SCENARIOS,
    FINANCIAL_PROMPT, GENERAL_PROMPT, VOICE_PROMPT, SAFETY_PROMPT, DEEP_PROMPT,
)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4000
OUTPUT = Path("data/synthetic_qa")
OUTPUT.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic()


def gen(prompt_text):
    for attempt in range(3):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = r.content[0].text.strip()
            if "```" in text:
                for p in text.split("```"):
                    p = p.strip()
                    if p.startswith("json"): p = p[4:].strip()
                    if p.startswith("["): text = p; break
            msgs = json.loads(text)
            if isinstance(msgs, list) and len(msgs) >= 2:
                return msgs
        except anthropic.RateLimitError:
            print(f"  rate limited, waiting 60s...")
            time.sleep(60)
        except Exception as e:
            if "credit" in str(e).lower():
                print(f"\n  OUT OF CREDITS"); sys.exit(1)
            print(f"  error: {str(e)[:80]}")
            time.sleep(5)
    return None


def run_category(name, task_list, outfile):
    data = []
    if outfile.exists():
        with open(outfile) as f: data = json.load(f)

    remaining = task_list[len(data):]
    total = len(task_list)
    print(f"\n=== {name}: {len(data)}/{total} done, {len(remaining)} remaining ===")

    for i, task in enumerate(remaining):
        idx = len(data) + 1
        print(f"  [{idx}/{total}] {task['topic'][:55]}...", end=" ", flush=True)

        msgs = gen(task["prompt"])
        if msgs:
            data.append({"topic": task["topic"], "profile": task["profile"],
                         "conversation": msgs, "category": name})
            with open(outfile, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"OK ({len(msgs)} msgs)")
        else:
            print("FAIL")

        time.sleep(45)  # rate limit: 8K tokens/min, ~4K per req

    print(f"  Done: {len(data)}/{total}")
    return data


def build_financial():
    tasks = []
    random.seed(42)
    for topic in FINANCIAL_TOPICS:
        for profile in PROFILES:
            tasks.append({
                "prompt": FINANCIAL_PROMPT.format(
                    topic=topic, scenario=random.choice(SCENARIOS),
                    name=profile["name"], gender=profile["gender"],
                    age=profile["age"], form=profile["form"],
                    turns=3, msg_count=6),
                "topic": topic,
                "profile": {"name": profile["name"], "gender": profile["gender"],
                            "age": profile["age"], "address_form": profile["form"]},
            })
    random.shuffle(tasks)
    return tasks


def build_general():
    tasks = []
    for topic in GENERAL_TOPICS:
        for profile in PROFILES[:8]:
            tasks.append({
                "prompt": GENERAL_PROMPT.format(
                    topic=topic, name=profile["name"],
                    gender=profile["gender"], age=profile["age"]),
                "topic": topic,
                "profile": {"name": profile["name"], "gender": profile["gender"],
                            "age": profile["age"]},
            })
    random.seed(43); random.shuffle(tasks)
    return tasks


def build_voice():
    tasks = []
    for topic in VOICE_USSD_TOPICS:
        for profile in PROFILES:
            tasks.append({
                "prompt": VOICE_PROMPT.format(
                    topic=topic, name=profile["name"], age=profile["age"]),
                "topic": topic,
                "profile": {"name": profile["name"], "age": profile["age"]},
            })
    random.seed(44); random.shuffle(tasks)
    return tasks


def build_safety():
    tasks = []
    for topic in SAFETY_TOPICS:
        for profile in PROFILES[:6]:
            tasks.append({
                "prompt": SAFETY_PROMPT.format(
                    topic=topic, name=profile["name"],
                    gender=profile["gender"], age=profile["age"],
                    form=profile["form"]),
                "topic": topic,
                "profile": {"name": profile["name"], "gender": profile["gender"],
                            "age": profile["age"]},
            })
    random.seed(45); random.shuffle(tasks)
    return tasks


def build_deep():
    tasks = []
    for topic in MULTI_TURN_TOPICS:
        for profile in PROFILES[:6]:
            tasks.append({
                "prompt": DEEP_PROMPT.format(
                    topic=topic, name=profile["name"],
                    gender=profile["gender"], age=profile["age"],
                    form=profile["form"]),
                "topic": topic,
                "profile": {"name": profile["name"], "gender": profile["gender"],
                            "age": profile["age"]},
            })
    random.seed(46); random.shuffle(tasks)
    return tasks


# Test API
print("Testing API...", end=" ")
try:
    r = client.messages.create(model=MODEL, max_tokens=20,
                                messages=[{"role": "user", "content": "ሰላም"}])
    print(f"OK: {r.content[0].text.strip()}")
except Exception as e:
    print(f"FAILED: {e}"); sys.exit(1)

cat = sys.argv[1] if len(sys.argv) > 1 else "financial"

if cat == "financial":
    run_category("financial", build_financial(), OUTPUT / "financial_conversations.json")
elif cat == "general":
    run_category("general", build_general(), OUTPUT / "general_conversations.json")
elif cat == "voice":
    run_category("voice", build_voice(), OUTPUT / "voice_conversations.json")
elif cat == "safety":
    run_category("safety", build_safety(), OUTPUT / "safety_conversations.json")
elif cat == "deep":
    run_category("deep", build_deep(), OUTPUT / "deep_conversations.json")
elif cat == "all":
    run_category("financial", build_financial(), OUTPUT / "financial_conversations.json")
    run_category("general", build_general(), OUTPUT / "general_conversations.json")
    run_category("voice", build_voice(), OUTPUT / "voice_conversations.json")
    run_category("safety", build_safety(), OUTPUT / "safety_conversations.json")
    run_category("deep", build_deep(), OUTPUT / "deep_conversations.json")
elif cat == "status":
    for name in ["financial", "general", "voice", "safety", "deep"]:
        f = OUTPUT / f"{name}_conversations.json"
        count = len(json.load(open(f))) if f.exists() else 0
        builders = {"financial": build_financial, "general": build_general,
                     "voice": build_voice, "safety": build_safety, "deep": build_deep}
        total = len(builders[name]())
        print(f"  {name:12s} {count:5d}/{total:5d}")
    legacy = OUTPUT / "raw_generated.json"
    if legacy.exists():
        print(f"  {'legacy':12s} {len(json.load(open(legacy))):5d}")
