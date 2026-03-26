"""
Generate 50K synthetic Amharic conversations for Tibeb production training.

Categories:
  - 20K financial conversations (core Tibeb domain)
  - 10K general Amharic Q&A (fluency)
  - 5K voice/USSD short-form (accessibility)
  - 5K safety & alignment (refusals, boundaries)
  - 5K multi-turn deep dives (complex topics)
  - 5K Ethiopian context (culture, regulations, daily life)

Respects API rate limits with configurable concurrency.
Saves progress after every batch — safe to interrupt and resume.

Usage:
    export ANTHROPIC_API_KEY=your_key
    python scripts/generate_50k.py                    # generate all
    python scripts/generate_50k.py --category financial --max 1000
    python scripts/generate_50k.py --status           # show progress
"""

import anthropic
import asyncio
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = Path("data/synthetic_qa")

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2000
CONCURRENCY = 1   # sequential — rate limit is 8K output tokens/min
BATCH_SIZE = 10    # save progress every N conversations
REQUEST_DELAY = 35 # seconds between requests to stay under rate limit


# =====================================================================
# PROFILES (12 diverse Ethiopian users)
# =====================================================================
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


# =====================================================================
# TOPIC DATABASES BY CATEGORY
# =====================================================================

FINANCIAL_TOPICS = [
    # T-Bills & Government Securities (20 topics)
    "T-Bill investment basics for absolute beginners",
    "How to participate in NBE T-Bill auction",
    "T-Bill vs bank fixed deposit — detailed comparison",
    "91-day vs 182-day vs 364-day T-Bill — which to choose",
    "Understanding T-Bill discount pricing and calculating yield",
    "Secondary market for T-Bills in Ethiopia",
    "T-Bill risks — what could go wrong",
    "How T-Bill interest rates affect the economy",
    "Government bonds beyond T-Bills in Ethiopia",
    "T-Bill minimum investment and denomination rules",
    "T-Bill auction results — how to read them",
    "Reinvesting T-Bill proceeds — rolling strategy",
    "T-Bill taxation in Ethiopia",
    "History of T-Bill rates in Ethiopia over past 5 years",
    "T-Bills during high inflation — are they still worth it",
    "Institutional vs individual T-Bill investing",
    "How to track your T-Bill investments",
    "T-Bill liquidity — can you sell before maturity",
    "T-Bill vs equb — traditional vs modern saving",
    "Teaching family about T-Bills — how to explain simply",
    # ESX & Equities (25 topics)
    "What is the Ethiopian Securities Exchange explained from scratch",
    "Steps to buy your first share on ESX — complete guide",
    "How to read stock prices and market data on ESX",
    "Understanding IPOs and how to participate in Ethiopia",
    "Ethio Telecom IPO — complete story and lessons",
    "How share prices change — supply, demand, and market forces",
    "Dividend payments — what they are and when to expect them",
    "Market capitalization explained with Ethiopian examples",
    "Reading a stock ticker and understanding price movements",
    "Stock market volatility — why prices go up and down",
    "Long-term vs short-term investing on ESX",
    "Sector analysis — which Ethiopian industries to watch",
    "How to evaluate a company before buying its shares",
    "P/E ratio and valuation basics for Ethiopian stocks",
    "Rights issues and stock splits — what shareholders need to know",
    "Market orders vs limit orders on ESX",
    "Building a diversified Ethiopian stock portfolio",
    "ESX trading hours and settlement process",
    "Corporate governance and shareholder rights in Ethiopia",
    "Following ESX news and announcements",
    "Common mistakes first-time Ethiopian stock investors make",
    "ESX vs international stock markets — key differences",
    "Ethiopian banking sector stocks analysis",
    "Insurance company stocks on ESX",
    "Manufacturing and industrial stocks on ESX",
    # Broker & Account (15 topics)
    "How to choose a licensed broker in Ethiopia",
    "Opening a broker account — step by step",
    "KYC requirements for Ethiopian investment accounts",
    "Online vs in-person broker account setup",
    "Minimum investment amounts across different Ethiopian brokers",
    "Broker fees and commissions in Ethiopia",
    "Switching brokers — how and when",
    "What to do if your broker closes",
    "Checking your portfolio and transaction history",
    "Understanding broker statements and reports",
    "Broker regulation by ECMA — your protections",
    "CSD (Central Securities Depository) account explained",
    "Joint accounts and investment — couples investing together",
    "Corporate investment accounts for businesses",
    "Diaspora investor accounts — special requirements",
    # Banking & Savings (20 topics)
    "Types of bank accounts in Ethiopia explained",
    "Fixed deposit vs savings account — detailed comparison",
    "Interest rates across Ethiopian banks — current comparison",
    "CBE vs private banks — pros and cons",
    "Mobile banking setup and security in Ethiopia",
    "Telebirr complete guide — registration to investing",
    "CBEBirr and other bank mobile money services",
    "Foreign currency accounts in Ethiopia",
    "Children's savings accounts in Ethiopia",
    "Business banking in Ethiopia — what entrepreneurs need",
    "Bank loan types in Ethiopia — personal, business, mortgage",
    "Understanding bank interest rate calculation",
    "Automatic savings plans at Ethiopian banks",
    "ATM and debit card usage in Ethiopia",
    "Online banking security — protecting your accounts",
    "Bank deposit insurance in Ethiopia — is your money safe",
    "Microfinance institutions vs commercial banks",
    "Women-focused banking products in Ethiopia",
    "Student banking and financial products",
    "Senior citizen banking benefits in Ethiopia",
    # FX & Currency (10 topics)
    "Understanding the Ethiopian Birr exchange rate",
    "Official vs parallel market exchange rates",
    "How NBE manages the Birr — monetary policy basics",
    "Impact of Birr devaluation on everyday life",
    "Sending remittances to Ethiopia — best practices",
    "Foreign currency regulations for Ethiopian residents",
    "Dollar-denominated savings options in Ethiopia",
    "Import/export business and FX management",
    "Travel and forex — getting foreign currency legally",
    "Cryptocurrency regulations in Ethiopia",
    # Inflation & Economics (15 topics)
    "Understanding inflation and its effect on your money",
    "CPI and inflation measurement in Ethiopia",
    "How to protect savings from inflation",
    "NBE interest rate decisions explained",
    "Monetary policy and its effect on everyday Ethiopians",
    "Cost of living in Addis Ababa vs regional cities",
    "Wage growth vs inflation in Ethiopia",
    "Food price inflation and household budgeting",
    "Real returns — adjusting investment gains for inflation",
    "Ethiopian GDP growth and what it means for investors",
    "Government fiscal policy and budget impact",
    "Trade deficit and balance of payments explained",
    "Ethiopian debt — domestic and foreign",
    "Economic reforms and liberalization in Ethiopia",
    "Impact of global economy on Ethiopian markets",
    # Risk & Strategy (15 topics)
    "Investment risk types — market, credit, liquidity, inflation",
    "Risk vs return tradeoff explained with Ethiopian examples",
    "Age-based investment strategy for Ethiopians",
    "Emergency fund — how much and where to keep it",
    "Portfolio diversification for Ethiopian context",
    "Rebalancing your portfolio — when and how",
    "Dollar cost averaging strategy in Ethiopian markets",
    "Lump sum vs periodic investing — which is better",
    "Investment horizon and goal-based investing",
    "Risk tolerance assessment — know yourself",
    "Common investment biases and how to avoid them",
    "When to sell — exit strategies for Ethiopian investors",
    "Inheritance and estate planning in Ethiopia",
    "Insurance as part of financial planning",
    "Investment record keeping and tax reporting",
    # Regulation (10 topics)
    "ECMA role and investor protection",
    "Your rights as an investor in Ethiopia",
    "Filing complaints against brokers or financial institutions",
    "Securities fraud and how to recognize it",
    "Insider trading laws in Ethiopia",
    "Capital gains tax calculation and filing",
    "Investment income tax in Ethiopia",
    "Anti-money laundering rules for investors",
    "Financial reporting requirements for companies",
    "New financial regulations and their impact",
    # Ethiopian-specific (20 topics)
    "Equb — traditional savings circles explained and modernized",
    "Edir — community mutual aid and its financial role",
    "Investing on a small salary — starting from 500 birr",
    "Ethiopian diaspora investment opportunities",
    "Rural Ethiopian access to financial services",
    "Women and investing in Ethiopia — barriers and opportunities",
    "Retirement planning with Ethiopian pension system",
    "Islamic finance options — Sharia-compliant investing in Ethiopia",
    "Real estate vs stock market in Ethiopian context",
    "Gold and precious metals investing in Ethiopia",
    "Agricultural investment — coffee, teff, sesame",
    "Livestock as investment in rural Ethiopia",
    "Starting a small business vs investing in markets",
    "Ethiopian commodity exchange (ECX) explained",
    "Cooperative societies and investment",
    "Youth entrepreneurship and financing options",
    "Government development bonds and savings certificates",
    "NGO and social enterprise investment in Ethiopia",
    "Education investment — ROI of higher education",
    "Healthcare costs and health savings in Ethiopia",
]

GENERAL_TOPICS = [
    # Ethiopian culture (20)
    "Ethiopian coffee ceremony — history and meaning",
    "Ethiopian calendar explained — how to convert dates",
    "Meskel celebration — traditions and significance",
    "Timket (Epiphany) celebration in Ethiopia",
    "Enkutatash — Ethiopian New Year traditions",
    "Ethiopian cuisine — injera, wot varieties, and regional dishes",
    "Ethiopian wedding traditions across different cultures",
    "Ethiopian music — from traditional to Ethio-jazz",
    "Ethiopian athletics — marathon and distance running legacy",
    "Historic sites — Lalibela, Axum, Gondar, Harar",
    "Ethiopian languages — Amharic, Oromo, Tigrinya, and more",
    "Ge'ez script — ancient writing system history",
    "Ethiopian Orthodox Church traditions",
    "Ethiopian fashion — traditional clothing and modern fusion",
    "Ethiopian art — painting, sculpture, and crafts",
    "Ethiopian literature and famous authors",
    "Coffee origins — Ethiopia as birthplace of coffee",
    "Ethiopian spices — berbere, mitmita, and their uses",
    "Fasting traditions in Ethiopian culture",
    "Ethiopian naming conventions and meaning",
    # Daily life (20)
    "Cooking basic Ethiopian dishes — doro wot recipe",
    "Ethiopian proverbs about wisdom and their meanings",
    "Job hunting in Addis Ababa — practical tips",
    "Education system in Ethiopia — from primary to university",
    "Healthcare — public vs private hospitals in Ethiopia",
    "Transportation in Addis — light rail, minibus, Ride app",
    "Renting an apartment in Addis Ababa — practical guide",
    "Starting a small business in Ethiopia — legal steps",
    "Ethiopian weather and seasons across regions",
    "Water and electricity services and billing",
    "Internet and mobile data in Ethiopia — costs and providers",
    "Ethiopian driving license and traffic rules",
    "Shopping in Ethiopia — markets vs supermarkets",
    "Ethiopian postal and delivery services",
    "Gym, sports, and fitness culture in Ethiopia",
    "Ethiopian social media and digital culture",
    "Volunteering and community service in Ethiopia",
    "Environmental issues in Ethiopia — conservation efforts",
    "Ethiopian parks and wildlife — tourist guide",
    "Moving to Addis Ababa from a regional city",
    # Knowledge (20)
    "How the internet works — simple explanation",
    "What is artificial intelligence — beginner guide",
    "Climate change and its effect on Ethiopia specifically",
    "Ethiopian agriculture — teff, coffee, and exports",
    "The GERD (Grand Ethiopian Renaissance Dam) explained",
    "Ethiopian history — from ancient Axum to modern era",
    "Ethiopian constitution and government structure",
    "Ethiopian diaspora communities worldwide",
    "Technology adoption in Ethiopia — growth and challenges",
    "Ethiopian Airlines — history and significance",
    "Space science in Ethiopia — the satellite program",
    "Ethiopian universities — top institutions and programs",
    "Ethiopian sports beyond running — football, volleyball",
    "Ethiopia and the African Union",
    "Ethiopian innovations and inventions",
    "Water resources and management in Ethiopia",
    "Ethiopian cities — Addis Ababa, Bahir Dar, Hawassa overview",
    "Ethiopian manufacturing and industrialization",
    "Tourism in Ethiopia — major attractions",
    "Ethiopian media and journalism",
]

VOICE_USSD_TOPICS = [
    # Short-form Q&A (designed for voice/USSD — brief responses)
    "What is today's T-Bill rate?",
    "How much is the dollar today?",
    "Is the stock market open today?",
    "How do I check my portfolio?",
    "What is ESX?",
    "What is a broker?",
    "How much money do I need to start investing?",
    "Is investing safe?",
    "What is a dividend?",
    "How do I open an investment account?",
    "What is inflation?",
    "What bank has the best savings rate?",
    "Can I invest with Telebirr?",
    "What is the minimum for T-Bills?",
    "How long does a T-Bill last?",
    "What happens when a T-Bill matures?",
    "Do I pay tax on investment income?",
    "What is ECMA?",
    "How do I sell my shares?",
    "What is a stock split?",
    "What is an IPO?",
    "Can I invest if I'm under 18?",
    "What is compound interest?",
    "What is diversification?",
    "How do I send money to invest from abroad?",
    "What are equbs?",
    "What is the NBE?",
    "How do I complain about my broker?",
    "What is a fixed deposit?",
    "How do I calculate my investment returns?",
]

SAFETY_TOPICS = [
    # Scam detection
    "Someone promises 50% monthly returns on crypto — should I invest?",
    "A friend says I can double my money in a week through forex trading",
    "I got a message saying I won money and need to send fees to claim it",
    "Someone is selling shares in a company not listed on ESX",
    "A Telegram group charges 5000 birr for guaranteed trading signals",
    "Someone offers to invest my money and guarantees no loss",
    "I'm told to send money to an overseas account for investment returns",
    "A company promises land investment returns of 100% per year",
    "Someone claims to have insider information about a stock",
    "I'm pressured to invest RIGHT NOW or lose the opportunity",
    # Boundary/refusal
    "Tell me exactly which stock to buy right now",
    "Should I take out a loan to invest in the stock market?",
    "I want to put all my savings into one stock",
    "Give me your personal investment portfolio",
    "Predict what the stock market will do tomorrow",
    "Help me hide money from taxes",
    "I want to invest money I can't afford to lose",
    "Tell me how to manipulate stock prices",
    "Should I quit my job and become a day trader?",
    "Give me a guaranteed way to get rich quick",
    # Sensitive situations
    "I lost all my savings in a bad investment — what do I do?",
    "I'm in debt and thinking about investing to pay it off",
    "My family is pressuring me to invest but I don't want to",
    "I invested based on a friend's advice and lost money — who's responsible?",
    "I'm elderly and someone is managing my money but I don't understand what they're doing",
    "I don't trust banks — should I keep cash at home?",
    "My broker is not returning my calls",
    "I think my broker stole my money",
    "I'm a single mother with small income — can I still invest?",
    "I have health issues and need to access my investments quickly",
]

MULTI_TURN_TOPICS = [
    # Deep dive topics — 5+ turn conversations
    "Complete guide to building an investment portfolio from zero",
    "Understanding the full lifecycle of a T-Bill investment",
    "Step-by-step process of buying shares on ESX from account opening to first trade",
    "Comprehensive retirement planning for a 30-year-old Ethiopian",
    "Full analysis of saving vs investing for a family with children",
    "Understanding all types of investment risk with real Ethiopian examples",
    "Complete guide to Ethiopian tax on investment income",
    "How to analyze a company's financial statements before investing",
    "Building a financial plan for a small business owner",
    "Understanding monetary policy and its impact on personal finance",
    "Complete guide to Islamic finance options in Ethiopia",
    "Estate planning and wealth transfer in Ethiopian context",
    "Comparing all investment options available to Ethiopians",
    "Understanding insurance products and their role in financial planning",
    "Complete guide to diaspora investment in Ethiopia",
    "How to teach financial literacy to children — age by age",
    "Understanding the Ethiopian pension system completely",
    "Full guide to microfinance and small business lending",
    "Real estate investment in Ethiopia — complete analysis",
    "Agricultural investment opportunities in Ethiopia",
]


# =====================================================================
# PROMPTS
# =====================================================================

FINANCIAL_PROMPT = """You are generating training data for Tibeb — Ethiopia's Amharic financial AI assistant.

Generate a realistic, natural conversation about: {topic}
Scenario: {scenario}

User profile:
- Name: {name}
- Gender: {gender}
- Age group: {age}
- Address form: {form}

Address form rules:
- male_informal: use አንተ, verb endings with ህ/ክ
- female_informal: use አንቺ, verb endings with ሽ
- formal_honorific: use እርስዎ, verb endings with ዎት/ቸው — ALWAYS for elders

CRITICAL AMHARIC QUALITY RULES:
- Write natural, flowing Amharic — NOT translated English
- Tibeb's responses must be detailed (150-250 words each), educational, warm
- Include specific Ethiopian examples: real ETB amounts, real institutions
- Correct spellings: ሰላም (not ሠላም), ሀገር (not ሃገር), እርስዎ (not እርሶ), አክሲዮን, ቲ-ቢል (hyphenated), ብር, ምንዛሬ, ኢትዮጵያ
- Arabic numerals for amounts. End sentences with ። Use ፣ for commas
- Tibeb NEVER gives direct investment advice — educational only, suggests professionals

Generate {turns} full exchanges ({msg_count} messages total).

Return ONLY a valid JSON array of message objects, no markdown:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]"""

GENERAL_PROMPT = """Generate a natural Amharic conversation about: {topic}

User: {name} ({gender}, {age})

Rules:
- Natural fluent Amharic, NOT translated English
- Informative responses (100-200 words each)
- Ethiopian context and examples
- Correct spellings: ሰላም, ሀገር, ኢትዮጵያ, እርስዎ
- Ethiopic punctuation: ። for period, ፣ for comma
- Arabic numerals for numbers

Generate 2 exchanges (4 messages).

Return ONLY valid JSON array:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]"""

VOICE_PROMPT = """Generate a SHORT Amharic voice/USSD interaction for Tibeb (Ethiopian financial AI).

User asks: {topic}
User: {name} ({age})

Rules:
- User question is 1-2 sentences max
- Tibeb response is 2-4 sentences max (BRIEF — suitable for voice or SMS)
- Natural Amharic, not translated English
- Accurate financial info
- Correct spellings: ሰላም, ቲ-ቢል, አክሲዮን, ብር

Generate 1 exchange (2 messages).

Return ONLY valid JSON array:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]"""

SAFETY_PROMPT = """Generate a training conversation where Tibeb (Ethiopian financial AI) handles a sensitive/safety scenario.

Scenario: {topic}
User: {name} ({gender}, {age}, address form: {form})

Rules:
- User asks something that requires careful handling
- Tibeb responds in warm, empathetic Amharic
- Tibeb REFUSES harmful requests politely but firmly
- Tibeb redirects to proper resources (ECMA, police, licensed advisor)
- For scams: Tibeb explains the red flags clearly
- For risky behavior: Tibeb explains why it's dangerous without being preachy
- Natural Amharic, correct spellings
- 2 exchanges (4 messages)

Return ONLY valid JSON array:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]"""

DEEP_PROMPT = """Generate a DEEP multi-turn Amharic conversation for Tibeb (Ethiopian financial AI).

Topic: {topic}
User: {name} ({gender}, {age}, address form: {form})

This should be a thorough exploration of the topic:
- User starts with a basic question, then goes deeper with each turn
- Tibeb provides increasingly detailed, specific answers
- Include real Ethiopian numbers, institutions, regulations
- Each Tibeb response should be 200-400 words
- Natural flowing Amharic, not translated English
- Correct spellings throughout

Generate 5 full exchanges (10 messages total).

Return ONLY valid JSON array:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]"""

SCENARIOS = [
    "beginner asking their very first question",
    "someone comparing options and needs help deciding",
    "someone who heard about this from a friend",
    "someone worried about risk and losing money",
    "someone investing for their children's future",
    "someone asking a follow-up after learning the basics",
    "a diaspora Ethiopian learning about local options",
    "a university student curious about finance",
    "a small business owner looking to diversify",
    "a retiree looking for safe income",
]


# =====================================================================
# GENERATION ENGINE
# =====================================================================

async def generate_one(client, prompt_text, semaphore):
    """Generate a single conversation with retry and rate limit handling."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                text = response.content[0].text.strip()

                if "```" in text:
                    for part in text.split("```"):
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("["):
                            text = part
                            break

                messages = json.loads(text)
                if isinstance(messages, list) and len(messages) >= 2:
                    # Respect rate limit between requests
                    await asyncio.sleep(REQUEST_DELAY)
                    return messages

            except json.JSONDecodeError:
                await asyncio.sleep(2)
            except anthropic.RateLimitError:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                if "credit" in str(e).lower():
                    print(f"\n  OUT OF CREDITS. Add more at console.anthropic.com")
                    return "CREDIT_ERROR"
                await asyncio.sleep(5 * (attempt + 1))

        return None


def build_tasks(category):
    """Build task list for a given category."""
    tasks = []
    random.seed(42)

    if category == "financial":
        for topic in FINANCIAL_TOPICS:
            for profile in PROFILES:
                scenario = random.choice(SCENARIOS)
                tasks.append({
                    "prompt": FINANCIAL_PROMPT.format(
                        topic=topic, scenario=scenario,
                        name=profile["name"], gender=profile["gender"],
                        age=profile["age"], form=profile["form"],
                        turns=3, msg_count=6,
                    ),
                    "topic": topic,
                    "profile": {"name": profile["name"], "gender": profile["gender"],
                                "age": profile["age"], "address_form": profile["form"]},
                    "category": "financial",
                })

    elif category == "general":
        for topic in GENERAL_TOPICS:
            for profile in PROFILES[:8]:
                tasks.append({
                    "prompt": GENERAL_PROMPT.format(
                        topic=topic, name=profile["name"],
                        gender=profile["gender"], age=profile["age"],
                    ),
                    "topic": topic,
                    "profile": {"name": profile["name"], "gender": profile["gender"],
                                "age": profile["age"]},
                    "category": "general",
                })

    elif category == "voice":
        for topic in VOICE_USSD_TOPICS:
            for profile in PROFILES:
                tasks.append({
                    "prompt": VOICE_PROMPT.format(
                        topic=topic, name=profile["name"], age=profile["age"],
                    ),
                    "topic": topic,
                    "profile": {"name": profile["name"], "age": profile["age"]},
                    "category": "voice",
                })

    elif category == "safety":
        for topic in SAFETY_TOPICS:
            for profile in PROFILES[:6]:
                tasks.append({
                    "prompt": SAFETY_PROMPT.format(
                        topic=topic, name=profile["name"],
                        gender=profile["gender"], age=profile["age"],
                        form=profile["form"],
                    ),
                    "topic": topic,
                    "profile": {"name": profile["name"], "gender": profile["gender"],
                                "age": profile["age"]},
                    "category": "safety",
                })

    elif category == "deep":
        for topic in MULTI_TURN_TOPICS:
            for profile in PROFILES[:6]:
                tasks.append({
                    "prompt": DEEP_PROMPT.format(
                        topic=topic, name=profile["name"],
                        gender=profile["gender"], age=profile["age"],
                        form=profile["form"],
                    ),
                    "topic": topic,
                    "profile": {"name": profile["name"], "gender": profile["gender"],
                                "age": profile["age"]},
                    "category": "deep",
                })

    random.shuffle(tasks)
    return tasks


def get_output_file(category):
    return OUTPUT_DIR / f"{category}_conversations.json"


def load_existing(category):
    path = get_output_file(category)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_data(category, data):
    path = get_output_file(category)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def generate_category(category, max_count=None):
    """Generate all conversations for a category."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    existing = load_existing(category)
    tasks = build_tasks(category)

    if max_count:
        tasks = tasks[:max_count]
    tasks = tasks[len(existing):]

    total = len(existing) + len(tasks)
    errors = 0
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  {category.upper()}: {len(existing)} existing, {len(tasks)} remaining, {total} target")
    print(f"  Concurrency: {CONCURRENCY}, Batch save: every {BATCH_SIZE}")
    print(f"{'='*60}\n")

    if not tasks:
        print("  Already complete!")
        return existing

    unsaved = 0
    for chunk_start in range(0, len(tasks), BATCH_SIZE):
        chunk = tasks[chunk_start:chunk_start + BATCH_SIZE]
        coros = [generate_one(client, t["prompt"], semaphore) for t in chunk]
        results = await asyncio.gather(*coros)

        credit_error = False
        for task, messages in zip(chunk, results):
            if messages == "CREDIT_ERROR":
                credit_error = True
                errors += 1
            elif isinstance(messages, list) and len(messages) >= 2:
                existing.append({
                    "topic": task["topic"],
                    "profile": task["profile"],
                    "conversation": messages,
                    "category": task["category"],
                })
            else:
                errors += 1

        save_data(category, existing)
        unsaved = 0

        elapsed = time.time() - start
        done = len(existing) - (total - len(tasks))
        rate = done / max(elapsed / 60, 0.1)
        remaining = len(tasks) - chunk_start - len(chunk)
        eta = remaining / max(rate, 0.1)

        print(f"  [{len(existing)}/{total}] {rate:.1f}/min | "
              f"{errors} errors | ETA {eta:.0f}min | {elapsed/60:.1f}min elapsed")

        if credit_error and errors > 5:
            print("\n  Too many errors (likely out of credits). Stopping.")
            print("  Progress saved. Resume by running again.")
            break

    return existing


def show_status():
    """Show generation progress across all categories."""
    categories = ["financial", "general", "voice", "safety", "deep"]
    total_convos = 0
    total_turns = 0

    print(f"\n{'='*60}")
    print(f"  Tibeb Data Generation Status")
    print(f"{'='*60}\n")

    for cat in categories:
        data = load_existing(cat)
        tasks = build_tasks(cat)
        turns = sum(len(x.get("conversation", [])) // 2 for x in data)
        total_convos += len(data)
        total_turns += turns
        pct = len(data) / max(len(tasks), 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cat:12s} {bar} {len(data):5d}/{len(tasks):5d} ({pct:.0f}%) | {turns} turns")

    # Also count legacy data
    legacy = Path("data/synthetic_qa/raw_generated.json")
    if legacy.exists():
        with open(legacy) as f:
            ld = json.load(f)
        lt = sum(len(x.get("conversation", [])) // 2 for x in ld)
        total_convos += len(ld)
        total_turns += lt
        print(f"  {'legacy':12s} {'█'*20} {len(ld):5d}/{len(ld):5d} (100%) | {lt} turns")

    print(f"\n  TOTAL: {total_convos} conversations, {total_turns} turn pairs")
    print(f"  With upsampling: ~{total_turns * 5} training rows")


async def main():
    parser = argparse.ArgumentParser(description="Generate 50K Tibeb conversations")
    parser.add_argument("--category", choices=["financial", "general", "voice", "safety", "deep", "all"],
                        default="all")
    parser.add_argument("--max", type=int, default=None, help="Max conversations per category")
    parser.add_argument("--status", action="store_true", help="Show progress only")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        show_status()
        return

    categories = ["financial", "general", "voice", "safety", "deep"] if args.category == "all" else [args.category]

    # Test API first
    print("Testing API...")
    try:
        client = anthropic.Anthropic()
        r = client.messages.create(model=MODEL, max_tokens=20,
                                    messages=[{"role": "user", "content": "ሰላም"}])
        print(f"  OK: {r.content[0].text.strip()}")
    except Exception as e:
        print(f"  API error: {e}")
        sys.exit(1)

    for cat in categories:
        await generate_category(cat, max_count=args.max)

    print("\n" + "="*60)
    show_status()


if __name__ == "__main__":
    asyncio.run(main())
