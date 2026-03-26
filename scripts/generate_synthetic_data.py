"""
Generate synthetic Amharic conversations for Tibeb training at scale.

Two modes:
  1. Financial conversations (Tibeb persona) — 50+ topics × 6 profiles × 3 scenarios
  2. General Amharic Q&A — teach natural conversational Amharic

Target: 5,000+ high-quality conversations.

Usage:
    export ANTHROPIC_API_KEY=your_key
    python scripts/generate_synthetic_data.py                    # generate all
    python scripts/generate_synthetic_data.py --mode financial   # financial only
    python scripts/generate_synthetic_data.py --mode general     # general only
    python scripts/generate_synthetic_data.py --max 500          # limit count
"""

import anthropic
import argparse
import json
import random
import time
from pathlib import Path

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Financial topics — massively expanded from original 10
# ---------------------------------------------------------------------------
FINANCIAL_TOPICS = [
    # T-Bills & Government Securities
    "T-Bill investment basics — what they are and how they work",
    "How to buy T-Bills through NBE auction",
    "T-Bill vs bank savings — which is better for beginners",
    "T-Bill maturity periods (91, 182, 364 days) and choosing the right one",
    "Understanding T-Bill discount pricing and yield calculation",
    "Secondary market for T-Bills in Ethiopia",
    # ESX & Equities
    "What is the Ethiopian Securities Exchange (ESX)",
    "How to read ESX share prices and market data",
    "Steps to buy your first share on ESX",
    "Understanding IPOs — how to participate in Ethiopia",
    "Ethio Telecom IPO — what happened and lessons learned",
    "How share prices go up and down — supply and demand",
    "Dividend payments — what they are and how they work",
    "Blue chip vs growth stocks in Ethiopian context",
    "Market capitalization and what it means",
    "Reading a stock ticker and price chart basics",
    # Broker & Account Setup
    "How to open a broker account in Ethiopia",
    "Choosing a licensed broker — what to look for",
    "KYC requirements for opening an investment account",
    "Online vs in-person broker account opening",
    "Minimum investment amounts in Ethiopia",
    # Banking & Savings
    "Ethiopian banking vs capital markets — key differences",
    "Types of bank accounts in Ethiopia (savings, checking, fixed)",
    "Fixed deposit vs T-Bill — comparing returns",
    "Interest rates on Ethiopian bank savings accounts",
    "Commercial Bank of Ethiopia vs private banks for savings",
    "Mobile banking in Ethiopia — Telebirr and bank apps",
    # FX & Currency
    "FX rate impact on ETB savings and purchasing power",
    "Understanding the parallel market vs official exchange rate",
    "How NBE manages the Ethiopian Birr exchange rate",
    "Impact of currency devaluation on everyday Ethiopians",
    "Sending remittances to Ethiopia — how FX affects family",
    # Inflation & Economics
    "Inflation impact on ETB savings — how money loses value",
    "NBE interest rate decisions and what they mean for you",
    "Cost of living increases in Addis Ababa",
    "How to protect savings from inflation in Ethiopia",
    "Understanding CPI and inflation measurement in Ethiopia",
    # Risk & Strategy
    "Portfolio diversification for Ethiopian investors",
    "Understanding investment risk — beginner explanation",
    "Risk vs return tradeoff explained simply",
    "Age-based investment strategy in Ethiopian context",
    "Emergency fund before investing — why it matters",
    "Don't put all eggs in one basket — Ethiopian examples",
    # Regulation & Protection
    "ECMA (Ethiopian Capital Markets Authority) investor protection",
    "Your rights as an investor in Ethiopia",
    "How to file a complaint against a broker",
    "Avoiding investment scams in Ethiopia",
    "Tax on investment income in Ethiopia",
    "Capital gains tax in Ethiopia — what you owe",
    # Specific Ethiopian Context
    "Investing with small amounts — starting from 1,000 birr",
    "Investment options for Ethiopian diaspora",
    "Rural Ethiopians and investment access",
    "Women and investing in Ethiopia — breaking barriers",
    "Retirement planning in Ethiopia — pension vs investing",
    "Islamic finance options in Ethiopia (Sharia-compliant)",
    "Real estate vs stock market in Ethiopia",
    "Gold and commodity investing in Ethiopian context",
    "Insurance products in Ethiopia — are they investments?",
    "Microfinance institutions and small business investing",
    # Telebirr & Digital Finance
    "Using Telebirr for investment transactions",
    "Digital payment and investment in Ethiopia",
    "Mobile money as a gateway to investing",
    # Education & Literacy
    "What is financial literacy and why it matters",
    "Teaching children about money in Ethiopian culture",
    "Budgeting basics for Ethiopian households",
    "Debt management — when borrowing makes sense",
    "Understanding compound interest with Ethiopian examples",
    "The time value of money explained in Amharic",
]

FINANCIAL_SCENARIOS = [
    "beginner asking their very first question",
    "someone comparing two options and needs help deciding",
    "someone who heard about this from a friend and is curious",
    "someone worried about losing money",
    "someone who wants to invest for their children's future",
    "someone asking a follow-up after a previous conversation",
]

# ---------------------------------------------------------------------------
# General Amharic conversation topics
# ---------------------------------------------------------------------------
GENERAL_TOPICS = [
    # Ethiopian culture & daily life
    "Ethiopian coffee ceremony tradition and its meaning",
    "Ethiopian calendar vs Gregorian — how to convert",
    "Major Ethiopian holidays — Meskel, Timket, Enkutatash",
    "Ethiopian cuisine — injera, doro wot, and regional dishes",
    "Wedding traditions in Ethiopia",
    "Ethiopian music genres — traditional, modern, Ethio-jazz",
    "Ethiopian runners and athletics achievements",
    "Historic sites of Ethiopia — Lalibela, Axum, Gondar",
    "Ethiopian languages and writing systems",
    "Ge'ez script history and modern Amharic",
    # Practical daily life
    "How to cook basic Ethiopian dishes",
    "Ethiopian proverbs and their meanings",
    "Job hunting in Addis Ababa — tips and advice",
    "Education system in Ethiopia — primary to university",
    "Healthcare system in Ethiopia — public vs private",
    "Transportation in Addis Ababa — light rail, minibus, ride-hailing",
    "Renting an apartment in Addis Ababa",
    "Starting a small business in Ethiopia",
    "Ethiopian seasons and weather patterns",
    "Water and electricity services in Ethiopia",
    # Knowledge & explanation
    "How does the internet work — explained simply",
    "What is artificial intelligence — simple explanation",
    "Climate change and its effect on Ethiopia",
    "Ethiopian agriculture — teff, coffee, and exports",
    "The Nile dam (GERD) — what it means for Ethiopia",
    "Ethiopian history — brief overview from Axum to modern era",
    "Democracy and governance in Ethiopia",
    "Ethiopian diaspora communities around the world",
    "Technology adoption in Ethiopia — mobile phones and internet",
    "Ethiopian airlines and its significance",
]

# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------
PROFILES = [
    {"gender": "male",   "age": "youth",  "form": "male_informal",    "name": "ዳዊት"},
    {"gender": "female", "age": "youth",  "form": "female_informal",  "name": "ሳራ"},
    {"gender": "male",   "age": "elder",  "form": "formal_honorific", "name": "አቶ ተስፋዬ"},
    {"gender": "female", "age": "elder",  "form": "formal_honorific", "name": "ወይዘሮ አልማዝ"},
    {"gender": "male",   "age": "adult",  "form": "male_informal",    "name": "ዮሐንስ"},
    {"gender": "female", "age": "adult",  "form": "female_informal",  "name": "ሂሩት"},
    {"gender": "male",   "age": "youth",  "form": "male_informal",    "name": "ቃልኪዳን"},
    {"gender": "female", "age": "adult",  "form": "female_informal",  "name": "መስከረም"},
    {"gender": "male",   "age": "adult",  "form": "male_informal",    "name": "ብርሃኑ"},
    {"gender": "female", "age": "elder",  "form": "formal_honorific", "name": "ወይዘሮ ጽጌ"},
    {"gender": "male",   "age": "elder",  "form": "formal_honorific", "name": "አቶ ገብረ"},
    {"gender": "female", "age": "youth",  "form": "female_informal",  "name": "ፍቅርተ"},
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FINANCIAL_PROMPT = """You are generating training data for Tibeb — Ethiopia's Amharic financial AI assistant.

Generate a realistic, natural conversation about: {topic}
Scenario: {scenario}

User profile:
- Name: {name}
- Gender: {gender}
- Age group: {age}
- Address form: {form}

Address form rules:
- male_informal: use አንተ, verb endings with ህ/ክ, greeting ሰላም or እንደምን ነህ
- female_informal: use አንቺ, verb endings with ሽ/ሽ, greeting ሰላም or እንደምን ነሽ
- formal_honorific: use እርስዎ, verb endings with ዎት/ቸው, greeting እንደምን ናቸው — ALWAYS for elders

MANDATORY AMHARIC QUALITY RULES:
- Write natural, flowing Amharic — NOT word-by-word translation from English
- Use Amharic sentence structure (SOV — Subject Object Verb)
- Use rich Amharic vocabulary, not just Amharicized English words
- Tibeb's responses should be detailed (100-300 words each), educational, warm
- Include specific Ethiopian examples: real ETB amounts, real institutions, real scenarios
- ሰላም not ሠላም, ሀገር not ሃገር, እርስዎ not እርሶ, አክሲዮን not አክስዮን
- ቲ-ቢል (hyphenated), ብር not ብርር, ምንዛሬ not ምንዛሪ, ኢትዮጵያ not ኢትዮጲያ
- Use Arabic numerals for amounts/percentages
- End sentences with ። Use ፣ for commas
- Tibeb NEVER gives direct investment advice — always educational, suggests consulting professionals

Generate 3 full exchanges (user, assistant, user, assistant, user, assistant) — 6 messages total.
The conversation should flow naturally, with follow-up questions building on previous answers.

Return ONLY valid JSON array of messages, no markdown:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]"""


GENERAL_PROMPT = """You are generating Amharic conversation training data.

Generate a natural Amharic Q&A conversation about: {topic}

User profile:
- Name: {name}
- Gender: {gender}
- Age group: {age}

MANDATORY QUALITY RULES:
- Write natural, fluent Amharic — NOT translated English
- Use proper Amharic sentence structure (SOV)
- Use rich Amharic vocabulary appropriate to the topic
- Responses should be informative and detailed (100-200 words each)
- Use correct Ethiopic punctuation: ። for period, ፣ for comma, ? for questions
- Include specific Ethiopian context and examples
- Correct spellings: ሰላም, ሀገር, ኢትዮጵያ, እርስዎ
- Use Arabic numerals for numbers

Generate 2 full exchanges (4 messages total).

Return ONLY valid JSON array of messages, no markdown:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]"""


def generate_conversation(prompt_text, max_retries=2):
    """Generate a single conversation, with retry on failure."""
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = response.content[0].text.strip()

            # Strip markdown code fences if present
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
            else:
                print(f"    Unexpected format (attempt {attempt+1})")

        except json.JSONDecodeError as e:
            print(f"    JSON error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"    Error (attempt {attempt+1}): {e}")
            if "rate" in str(e).lower():
                time.sleep(5)

        time.sleep(1)

    return None


def generate_financial_data(output_file, max_count=None):
    """Generate financial conversations at scale."""
    all_convos = []
    if Path(output_file).exists():
        with open(output_file, encoding="utf-8") as f:
            all_convos = json.load(f)
        print(f"Resuming from {len(all_convos)} existing conversations...")

    # Generate all combinations
    combos = []
    for topic in FINANCIAL_TOPICS:
        for profile in PROFILES:
            scenario = random.choice(FINANCIAL_SCENARIOS)
            combos.append((topic, profile, scenario))

    random.seed(42)
    random.shuffle(combos)

    if max_count:
        combos = combos[:max_count]

    # Skip already generated (approximate by count)
    combos = combos[len(all_convos):]

    total = len(all_convos) + len(combos)
    errors = 0

    print(f"Generating {len(combos)} financial conversations (total target: {total})...\n")

    for i, (topic, profile, scenario) in enumerate(combos):
        idx = len(all_convos) + 1
        print(f"  [{idx}/{total}] {profile['name']} — {topic[:50]}...")

        prompt_text = FINANCIAL_PROMPT.format(
            topic=topic,
            scenario=scenario,
            name=profile["name"],
            gender=profile["gender"],
            age=profile["age"],
            form=profile["form"],
        )

        messages = generate_conversation(prompt_text)

        if messages:
            all_convos.append({
                "topic": topic,
                "profile": {
                    "name": profile["name"],
                    "gender": profile["gender"],
                    "age": profile["age"],
                    "address_form": profile["form"],
                },
                "conversation": messages,
                "type": "financial",
            })
            # Save after every conversation
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_convos, f, ensure_ascii=False, indent=2)
        else:
            errors += 1
            print(f"    FAILED")

        # Rate limiting
        time.sleep(0.3)

    print(f"\nFinancial: {len(all_convos)} conversations, {errors} errors")
    return all_convos


def generate_general_data(output_file, max_count=None):
    """Generate general Amharic conversations."""
    all_convos = []
    if Path(output_file).exists():
        with open(output_file, encoding="utf-8") as f:
            all_convos = json.load(f)
        print(f"Resuming from {len(all_convos)} existing conversations...")

    combos = []
    for topic in GENERAL_TOPICS:
        for profile in PROFILES[:6]:  # Use 6 profiles for general
            combos.append((topic, profile))

    random.seed(43)
    random.shuffle(combos)

    if max_count:
        combos = combos[:max_count]

    combos = combos[len(all_convos):]
    total = len(all_convos) + len(combos)
    errors = 0

    print(f"Generating {len(combos)} general conversations (total target: {total})...\n")

    for i, (topic, profile) in enumerate(combos):
        idx = len(all_convos) + 1
        print(f"  [{idx}/{total}] {profile['name']} — {topic[:50]}...")

        prompt_text = GENERAL_PROMPT.format(
            topic=topic,
            name=profile["name"],
            gender=profile["gender"],
            age=profile["age"],
        )

        messages = generate_conversation(prompt_text)

        if messages:
            all_convos.append({
                "topic": topic,
                "profile": {
                    "name": profile["name"],
                    "gender": profile["gender"],
                    "age": profile["age"],
                },
                "conversation": messages,
                "type": "general",
            })
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_convos, f, ensure_ascii=False, indent=2)
        else:
            errors += 1
            print(f"    FAILED")

        time.sleep(0.3)

    print(f"\nGeneral: {len(all_convos)} conversations, {errors} errors")
    return all_convos


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Amharic conversations for Tibeb")
    parser.add_argument("--mode", choices=["financial", "general", "all"],
                        default="all", help="Which type to generate")
    parser.add_argument("--max", type=int, default=None,
                        help="Max conversations per mode")
    args = parser.parse_args()

    Path("data/synthetic_qa").mkdir(parents=True, exist_ok=True)

    if args.mode in ("financial", "all"):
        financial = generate_financial_data(
            "data/synthetic_qa/raw_generated.json",
            max_count=args.max,
        )
        print(f"\nTotal financial conversations: {len(financial)}")

    if args.mode in ("general", "all"):
        general = generate_general_data(
            "data/synthetic_qa/general_amharic.json",
            max_count=args.max,
        )
        print(f"\nTotal general conversations: {len(general)}")


if __name__ == "__main__":
    main()
