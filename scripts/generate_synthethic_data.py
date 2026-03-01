import anthropic
import json
import time
from pathlib import Path

client = anthropic.Anthropic()

TOPICS = [
    "T-Bill investment basics",
    "ESX share prices and how to read them",
    "FX rate impact on ETB savings",
    "How to open a broker account in Ethiopia",
    "Understanding yield and interest calculations",
    "Portfolio diversification for Ethiopian investors",
    "ECMA investor protection rules",
    "Ethiopian banking vs capital markets",
    "Inflation impact on ETB savings",
    "NBE interest rate decisions and what they mean",
]

PROFILES = [
    {"gender": "male",   "age": "youth",  "form": "male_informal",    "name": "ዳዊት"},
    {"gender": "female", "age": "youth",  "form": "female_informal",  "name": "ሳራ"},
    {"gender": "male",   "age": "elder",  "form": "formal_honorific", "name": "አቶ ተስፋዬ"},
    {"gender": "female", "age": "elder",  "form": "formal_honorific", "name": "ወይዘሮ አልማዝ"},
    {"gender": "male",   "age": "adult",  "form": "male_informal",    "name": "ዮሐንስ"},
    {"gender": "female", "age": "adult",  "form": "female_informal",  "name": "ሂሩት"},
]

PROMPT = """You are generating training data for Tibeb — Ethiopia's first Amharic financial AI.

Generate a realistic conversation about: {topic}

User profile:
- Name: {name}
- Gender: {gender}
- Age group: {age}
- Address form: {form}

Address form rules:
- male_informal: use አንተ, verb endings with ህ, greeting እንደምን ነህ
- female_informal: use አንቺ, verb endings with ሽ, greeting እንደምን ነሽ
- formal_honorific: use እርስዎ, greeting እንደምን ናቸዉ — ALWAYS for elders

MANDATORY SPELLING RULES — never deviate from these:
- ሰላም not ሠላም
- ሰው not ሠው
- ሀገር not ሃገር
- እርስዎ not እርሶ
- አክሲዮን not አክስዮን
- ቲ-ቢል with hyphen always
- ብር not ብርር
- ምንዛሬ not ምንዛሪ
- ኢትዮጵያ not ኢትዮጲያ
- Use Arabic numerals for all numbers and percentages
- End Amharic sentences with ።
- Use ፣ for commas inside Amharic text

Requirements:
- User message in natural conversational Amharic, not stiff or formal
- Tibeb response in correct Amharic with proper honorifics for this profile
- Include real Ethiopian context: ETB amounts, ESX tickers, NBE rates
- Tibeb is warm, educational, never gives direct investment advice
- 2 full exchanges (user, assistant, user, assistant)

Return ONLY valid JSON, no markdown, no extra text:
{{"topic": "{topic}", "profile": {{"name": "{name}", "gender": "{gender}", "age": "{age}", "address_form": "{form}"}}, "conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "honorific_notes": "explain what form was used and why", "cultural_notes": "any Ethiopian cultural context"}}"""


def generate_pair(topic, profile):
    try:
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2000,
            messages=[{"role": "user", "content": PROMPT.format(
                topic=topic,
                name=profile["name"],
                gender=profile["gender"],
                age=profile["age"],
                form=profile["form"],
            )}]
        )
        text = response.content[0].text.strip()

        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    Path("data/synthetic_qa").mkdir(parents=True, exist_ok=True)
    output_file = "data/synthetic_qa/raw_generated.json"

    all_pairs = []
    if Path(output_file).exists():
        with open(output_file, encoding="utf-8") as f:
            all_pairs = json.load(f)
        print(f"Resuming from {len(all_pairs)} existing pairs...")

    errors = 0
    total = len(TOPICS) * len(PROFILES)
    print(f"Generating {total} conversations total...\n")

    for topic in TOPICS:
        for profile in PROFILES:
            print(f"  [{len(all_pairs)+1}/{total}] {profile['name']} ({profile['age']}) — {topic[:45]}...")

            pair = generate_pair(topic, profile)

            if pair:
                all_pairs.append(pair)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_pairs, f, ensure_ascii=False, indent=2)
                print(f"  ✓ Saved")
            else:
                errors += 1
                print(f"  ✗ Failed")

            time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Done.")
    print(f"Generated: {len(all_pairs)} conversations")
    print(f"Errors:    {errors}")
    print(f"Saved to:  {output_file}")


if __name__ == "__main__":
    main()






