"""
Collect human preference data for DPO training.

Workflow:
  1. Generate candidate responses from the fine-tuned model
  2. Present pairs to the human rater
  3. Save chosen/rejected pairs in DPO format

Usage:
    # Step 1: Generate candidate responses (requires trained adapter)
    python scripts/collect_preferences.py generate --num-prompts 200

    # Step 2: Rate pairs interactively
    python scripts/collect_preferences.py rate

    # Step 3: Export for DPO training
    python scripts/collect_preferences.py export
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROMPTS_FILE = "data/dpo/prompts.jsonl"
CANDIDATES_FILE = "data/dpo/candidates.jsonl"
PREFERENCES_FILE = "data/dpo/preferences.jsonl"
EXPORT_FILE = "data/dpo/dpo_train.jsonl"

# Amharic financial prompts for generating preference pairs
SEED_PROMPTS = [
    "ቲ-ቢል ምንድነው? እንዴት መግዛት እችላለሁ?",
    "አክሲዮን ገበያ ላይ ኢንቨስት ማድረግ እፈልጋለሁ። ከየት ልጀምር?",
    "ብሔራዊ ባንክ ወለድ ተመን ሲቀይር ምን ማለት ነው?",
    "የምንዛሬ ተመን ለቁጠባዬ ምን ተጽዕኖ አለው?",
    "ኢትዮ ቴሌኮም አክሲዮን እንዴት ልግዛ?",
    "ለጀማሪ ኢንቨስተር ምን ትመክራለህ?",
    "ቁጠባ ባንክ ውስጥ ማስቀመጥ ወይስ ኢንቨስት ማድረግ ይሻላል?",
    "የኢትዮጵያ ሴኩሪቲስ ኤክስቼንጅ ምንድነው?",
    "ዲቪደንድ ምንድነው? እንዴት ይከፈላል?",
    "ብሮከር ድርጅት እንዴት እመርጣለሁ?",
    "ፖርትፎሊዮ ዳይቨርሲፊኬሽን ማለት ምን ማለት ነው?",
    "የኢንፍሌሽን ተጽዕኖ ከቁጠባ ላይ ምንድነው?",
    "ቦንድ እና አክሲዮን ልዩነታቸው ምንድነው?",
    "ገንዘብ ማጠራቀም ለጀማሪ — ምን ምክር አለህ?",
    "ECMA ለኢንቨስተሮች ምን ጥበቃ ያደርጋል?",
    "ሰላም። ዛሬ የአክሲዮን ዋጋ ስንት ነው?",
    "ወለድ ላይ ያለ ግብር ስንት ነው?",
    "ለባለቤትነት ድርሻ ምን ያህል ገንዘብ ያስፈልጋል?",
    "የካፒታል ገበያ ምንድነው? ከቀላል ቁጠባ እንዴት ይለያል?",
    "ቴሌብር ተጠቅሜ አክሲዮን መግዛት እችላለሁ?",
    "የአክሲዮን ገበያ ሪስክ ምንድነው?",
    "ለልጆቼ ወደፊት ብዬ ኢንቨስት ማድረግ እፈልጋለሁ።",
    "ሰላም፣ ወይዘሮ ነኝ፣ ስለ ቲ-ቢል ማወቅ እፈልጋለሁ።",
    "አቶ ነኝ፣ ጡረታ ከወጣሁ በኋላ ገንዘቤን እንዴት ላስተዳድር?",
    "የውጭ ምንዛሬ ላይ ኢንቨስት ማድረግ ይቻላል?",
    "ስለ IPO ምን ማወቅ አለብኝ?",
    "ኢትዮጵያ ውስጥ ሙቹዋል ፈንድ አለ?",
    "ባንክ ውስጥ ያለኝ ቁጠባ ከኢንፍሌሽን ጋር ሲነጻጸር ምን ይሆናል?",
    "አነስተኛ ገቢ ያለኝ ሰው ኢንቨስት ማድረግ እችላለሁ?",
    "ስለ ኢትዮጵያ የካፒታል ገበያ ታሪክ ንገረኝ።",
]


def ensure_dirs():
    Path("data/dpo").mkdir(parents=True, exist_ok=True)


def generate_prompts(num_prompts):
    """Generate evaluation prompts from seed + training data."""
    ensure_dirs()
    prompts = list(SEED_PROMPTS)

    # Also pull prompts from the training data if available
    train_file = Path("data/tibeb_unified_train.jsonl")
    if train_file.exists():
        print("Sampling additional prompts from training data...")
        train_prompts = []
        with open(train_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("source") == "tibeb_financial" and row.get("instruction"):
                    train_prompts.append(row["instruction"])

        random.seed(42)
        if train_prompts:
            extra = random.sample(train_prompts, min(len(train_prompts), num_prompts - len(prompts)))
            prompts.extend(extra)

    prompts = prompts[:num_prompts]

    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({
                "id": i,
                "prompt": prompt,
            }, ensure_ascii=False) + "\n")

    print(f"Saved {len(prompts)} prompts to {PROMPTS_FILE}")
    return prompts


def generate_candidates(model_path=None, num_responses=3):
    """Generate multiple candidate responses per prompt using the model."""
    ensure_dirs()

    if not Path(PROMPTS_FILE).exists():
        print(f"Error: {PROMPTS_FILE} not found. Run 'generate' first.")
        sys.exit(1)

    prompts = []
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    # Try MLX first, then PyTorch
    adapter_path = model_path
    if not adapter_path:
        mlx_path = Path("models/tibeb-sft/mlx-adapter")
        pytorch_path = Path("models/tibeb-sft/pytorch-adapter")
        if mlx_path.exists():
            adapter_path = str(mlx_path)
        elif pytorch_path.exists():
            adapter_path = str(pytorch_path)

    if adapter_path and Path(adapter_path).exists():
        print(f"Using adapter: {adapter_path}")
        candidates = _generate_with_model(prompts, adapter_path, num_responses)
    else:
        print("No trained adapter found. Generating with base model variations.")
        print("(For better results, train the model first with finetune_tibeb.py)")
        candidates = _generate_with_api(prompts, num_responses)

    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Saved {len(candidates)} candidate sets to {CANDIDATES_FILE}")


def _generate_with_model(prompts, adapter_path, num_responses):
    """Generate responses using the local fine-tuned model."""
    import platform

    candidates = []

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        try:
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler

            adapter_config = Path(adapter_path) / "adapter_config.json"
            if adapter_config.exists():
                with open(adapter_config) as f:
                    config = json.load(f)
                model_id = config.get("model", "CohereForAI/aya-expanse-8b")
            else:
                model_id = "CohereForAI/aya-expanse-8b"

            print(f"Loading {model_id} with adapter...")
            model, tokenizer = load(model_id, adapter_path=adapter_path)

            for item in prompts:
                responses = []
                for temp in [0.3, 0.7, 1.0][:num_responses]:
                    try:
                        messages = [{"role": "user", "content": item["prompt"]}]
                        prompt_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True)
                        sampler = make_sampler(temp=temp)
                        response = generate(model, tokenizer, prompt=prompt_text,
                                            max_tokens=512, sampler=sampler)
                        responses.append({
                            "text": response,
                            "temperature": temp,
                        })
                    except Exception as e:
                        print(f"  Error generating for prompt {item['id']}: {e}")
                        responses.append({"text": f"[Error: {e}]", "temperature": temp})

                candidates.append({
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "responses": responses,
                })
                print(f"  [{item['id']+1}/{len(prompts)}] Generated {len(responses)} candidates")

            return candidates

        except ImportError:
            print("MLX not available, falling back to API generation")

    return _generate_with_api(prompts, num_responses)


def _generate_with_api(prompts, num_responses):
    """Generate diverse responses using Claude API as a proxy."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception:
        print("Error: No local model and no Anthropic API key.")
        print("Either train a model first or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    candidates = []

    system_prompt = """You are Tibeb — Ethiopia's Amharic financial AI assistant.
Rules:
- Respond in natural Amharic
- Use proper honorifics (እርስዎ for formal, አንተ/አንቺ for informal)
- Be warm, educational, never give direct investment advice
- Use correct spellings: ሰላም, ቲ-ቢል, አክሲዮን, ኢትዮጵያ, ብር, ምንዛሬ
- End sentences with ። and use ፣ for commas
- Use Arabic numerals for amounts and percentages"""

    temperatures = [0.3, 0.7, 1.0][:num_responses]

    for item in prompts:
        responses = []
        for temp in temperatures:
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    temperature=temp,
                    system=system_prompt,
                    messages=[{"role": "user", "content": item["prompt"]}],
                )
                responses.append({
                    "text": response.content[0].text,
                    "temperature": temp,
                })
            except Exception as e:
                print(f"  API error: {e}")
                responses.append({"text": f"[Error: {e}]", "temperature": temp})

            time.sleep(0.3)

        candidates.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "responses": responses,
        })
        print(f"  [{item['id']+1}/{len(prompts)}] Generated {len(responses)} candidates")

    return candidates


def rate_preferences():
    """Interactive CLI for rating response pairs."""
    ensure_dirs()

    if not Path(CANDIDATES_FILE).exists():
        print(f"Error: {CANDIDATES_FILE} not found. Run 'generate' first.")
        sys.exit(1)

    candidates = []
    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    # Load existing preferences to resume
    existing = set()
    preferences = []
    if Path(PREFERENCES_FILE).exists():
        with open(PREFERENCES_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pref = json.loads(line)
                    preferences.append(pref)
                    existing.add(pref["id"])
        print(f"Resuming from {len(preferences)} existing ratings...")

    print(f"\n{'='*60}")
    print(f"  Tibeb DPO Preference Rating")
    print(f"  Rate which response is better for each prompt.")
    print(f"  Enter: 1, 2, 3 (best response), 's' to skip, 'q' to quit")
    print(f"{'='*60}\n")

    rated = 0
    for item in candidates:
        if item["id"] in existing:
            continue

        responses = item["responses"]
        if len(responses) < 2:
            continue

        print(f"\n{'─'*60}")
        print(f"  Prompt [{item['id']+1}/{len(candidates)}]:")
        print(f"  {item['prompt']}")
        print(f"{'─'*60}")

        for i, resp in enumerate(responses):
            text = resp["text"]
            if text.startswith("[Error"):
                continue
            print(f"\n  Response {i+1} (temp={resp['temperature']}):")
            # Indent response text
            for line in text.split("\n"):
                print(f"    {line}")

        print()
        choice = input("  Best response (1/2/3), 's' to skip, 'q' to quit: ").strip().lower()

        if choice == "q":
            print("\nSaving and quitting...")
            break
        elif choice == "s":
            continue
        elif choice in ("1", "2", "3"):
            chosen_idx = int(choice) - 1
            if chosen_idx >= len(responses):
                print("  Invalid choice, skipping.")
                continue

            # Create preference pairs: chosen vs each rejected
            chosen = responses[chosen_idx]["text"]
            for i, resp in enumerate(responses):
                if i != chosen_idx and not resp["text"].startswith("[Error"):
                    preferences.append({
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "chosen": chosen,
                        "rejected": resp["text"],
                    })

            rated += 1
            # Save after each rating
            with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
                for pref in preferences:
                    f.write(json.dumps(pref, ensure_ascii=False) + "\n")
        else:
            print("  Invalid input, skipping.")

    print(f"\nRated {rated} prompts this session.")
    print(f"Total preference pairs: {len(preferences)}")
    print(f"Saved to: {PREFERENCES_FILE}")


def export_dpo():
    """Export preferences to DPO training format."""
    ensure_dirs()

    if not Path(PREFERENCES_FILE).exists():
        print(f"Error: {PREFERENCES_FILE} not found. Run 'rate' first.")
        sys.exit(1)

    preferences = []
    with open(PREFERENCES_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preferences.append(json.loads(line))

    if not preferences:
        print("No preferences found.")
        return

    # Convert to DPO format expected by trl
    dpo_data = []
    for pref in preferences:
        dpo_data.append({
            "prompt": pref["prompt"],
            "chosen": [
                {"role": "user", "content": pref["prompt"]},
                {"role": "assistant", "content": pref["chosen"]},
            ],
            "rejected": [
                {"role": "user", "content": pref["prompt"]},
                {"role": "assistant", "content": pref["rejected"]},
            ],
        })

    with open(EXPORT_FILE, "w", encoding="utf-8") as f:
        for row in dpo_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exported {len(dpo_data)} DPO pairs to {EXPORT_FILE}")
    print(f"Next: run 'python finetune_tibeb.py --dpo' to train with DPO")


def main():
    parser = argparse.ArgumentParser(description="Collect preference data for DPO")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate prompts and candidate responses")
    gen.add_argument("--num-prompts", type=int, default=200)
    gen.add_argument("--num-responses", type=int, default=3)
    gen.add_argument("--model-path", type=str, default=None)

    sub.add_parser("rate", help="Rate response pairs interactively")
    sub.add_parser("export", help="Export preferences to DPO format")

    args = parser.parse_args()

    if args.command == "generate":
        prompts = generate_prompts(args.num_prompts)
        generate_candidates(args.model_path, args.num_responses)
    elif args.command == "rate":
        rate_preferences()
    elif args.command == "export":
        export_dpo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
