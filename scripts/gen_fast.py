"""Fast async generator — uses full rate limit (90K output tokens/min, 1K req/min)."""
import anthropic
import asyncio
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
CONCURRENCY = 20  # 20 parallel requests — well within 1K req/min
OUTPUT = Path("data/synthetic_qa")
OUTPUT.mkdir(parents=True, exist_ok=True)

client = anthropic.AsyncAnthropic()
sem = asyncio.Semaphore(CONCURRENCY)


async def gen(prompt_text, idx, total):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.messages.create(
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
                wait = 30 * (attempt + 1)
                print(f"  [{idx}/{total}] rate limited, waiting {wait}s...", flush=True)
                await asyncio.sleep(wait)
            except json.JSONDecodeError:
                await asyncio.sleep(2)
            except Exception as e:
                if "credit" in str(e).lower():
                    print(f"\n  OUT OF CREDITS at [{idx}/{total}]", flush=True)
                    return "CREDIT_ERROR"
                await asyncio.sleep(3 * (attempt + 1))
        return None


async def run_category(name, task_list, outfile):
    data = []
    if outfile.exists():
        with open(outfile) as f:
            data = json.load(f)

    remaining = task_list[len(data):]
    total = len(task_list)
    start_count = len(data)
    print(f"\n=== {name}: {start_count}/{total} done, {len(remaining)} remaining ===", flush=True)

    if not remaining:
        print(f"  Already complete!", flush=True)
        return data

    # Process in batches of CONCURRENCY * 2 to save progress regularly
    batch_size = CONCURRENCY * 2
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_offset = start_count + batch_start

        tasks = []
        for i, task in enumerate(batch):
            idx = batch_offset + i + 1
            tasks.append(gen(task["prompt"], idx, total))

        results = await asyncio.gather(*tasks)

        credit_error = False
        new_count = 0
        for task, msgs in zip(batch, results):
            if msgs == "CREDIT_ERROR":
                credit_error = True
                break
            elif isinstance(msgs, list) and len(msgs) >= 2:
                data.append({
                    "topic": task["topic"],
                    "profile": task["profile"],
                    "conversation": msgs,
                    "category": name,
                })
                new_count += 1

        # Save after each batch
        with open(outfile, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  [{len(data)}/{total}] +{new_count} conversations saved", flush=True)

        if credit_error:
            print(f"\n  CREDITS EXHAUSTED. {len(data)}/{total} saved.", flush=True)
            return data

    print(f"  Done: {len(data)}/{total}", flush=True)
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


BUILDERS = {
    "financial": (build_financial, "financial_conversations.json"),
    "general": (build_general, "general_conversations.json"),
    "voice": (build_voice, "voice_conversations.json"),
    "safety": (build_safety, "safety_conversations.json"),
    "deep": (build_deep, "deep_conversations.json"),
}


async def main():
    # Test API
    print("Testing API...", end=" ", flush=True)
    try:
        c = anthropic.Anthropic()
        r = c.messages.create(model=MODEL, max_tokens=20,
                              messages=[{"role": "user", "content": "ሰላም"}])
        print(f"OK: {r.content[0].text.strip()}", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        sys.exit(1)

    cat = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cat == "status":
        for name, (builder, fname) in BUILDERS.items():
            f = OUTPUT / fname
            count = len(json.load(open(f))) if f.exists() else 0
            total = len(builder())
            print(f"  {name:12s} {count:5d}/{total:5d}")
        return

    if cat == "all":
        for name, (builder, fname) in BUILDERS.items():
            await run_category(name, builder(), OUTPUT / fname)
    elif cat in BUILDERS:
        builder, fname = BUILDERS[cat]
        await run_category(cat, builder(), OUTPUT / fname)
    else:
        print(f"Unknown category: {cat}. Use: {', '.join(BUILDERS.keys())}, all, status")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes", flush=True)
