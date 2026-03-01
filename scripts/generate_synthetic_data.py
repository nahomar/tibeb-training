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
- formal_honorific: use እርስዎ, greeting እንደምን ናቸዉ (ALWAYS for elders)

Requirements:
- User message in natural conversational Amharic
- Tibeb response in correct Amharic with proper honorifics for this profile
- Include real Ethiopian context: ETB amounts, ESX tickers, NBE rates
- Tibeb is educational, never gives direct investment advice
- 2 exchanges (user, assistant, user, assistant)

Return ONLY valid JSON:
{{"topic": "{topic}", "profile": {{"name": "{name}", "gender": "{gender}", "age": "{age}", "address_form": "{form}"}}, "conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], "honorific_notes": "explain the form used and why", "cultural_notes": "any Ethiopian cultural context"}}"""


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
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    Path("data/synthetic_qa").mkdir(parents=True, exist_ok=True)
    all_pairs = []
    errors = 0
    total = len(TOPICS) * len(PROFILES)
    print(f"Generating {total} conversations...")

    for topic in TOPICS:
        for profile in PROFILES:
            print(f"  {profile['name']} — {topic[:40]}...")
            pair = generate_pair(topic, profile)
            if pair:
                all_pairs.append(pair)
                with open("data/synthetic_qa/raw_generated.json", "w", encoding="utf-8") as f:
                    json.dump(all_pairs, f, ensure_ascii=False, indent=2)
            else:
                errors += 1
            time.sleep(0.5)

    print(f"Done. Generated: {len(all_pairs)}, Errors: {errors}")


if __name__ == "__main__":
    main()
