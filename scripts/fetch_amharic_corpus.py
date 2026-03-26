"""
Fetch native Amharic text corpora for continued pretraining / instruction tuning.

Sources:
  1. Amharic Wikipedia — clean encyclopedic text
  2. CC-100 Amharic — web-crawled Amharic text (Common Crawl)
  3. OSCAR Amharic — filtered web text
  4. Amharic News Corpus — news articles

Usage:
    python scripts/fetch_amharic_corpus.py
    python scripts/fetch_amharic_corpus.py --source wiki --max-rows 50000
    python scripts/fetch_amharic_corpus.py --source all --max-rows 100000
"""

import argparse
import json
import random
import re
from pathlib import Path

OUTPUT_DIR = Path("data/native_amharic")

CONTINUATION_TEMPLATES = [
    "ይህን ጽሑፍ ቀጥል።",
    "የሚከተለውን ዓረፍተ ነገር ጨርስ።",
    "ይህን ሀሳብ አስፋ።",
]


def clean_text(text):
    """Clean raw text: remove markup, normalize whitespace."""
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"={2,}[^=]+={2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def has_amharic(text, min_ratio=0.3):
    """Check if text has enough Amharic characters (Ethiopic script)."""
    if not text:
        return False
    ethiopic = sum(1 for c in text if "\u1200" <= c <= "\u137F" or "\u1380" <= c <= "\u139F")
    total = sum(1 for c in text if not c.isspace())
    if total == 0:
        return False
    return (ethiopic / total) >= min_ratio


def split_into_chunks(text, min_chars=100, max_chars=800):
    """Split text into sentence-bounded chunks suitable for training."""
    chunks = []
    paragraphs = text.split("\n\n")

    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if len(current) >= min_chars:
                chunks.append(current.strip())
            current = para

    if len(current) >= min_chars:
        chunks.append(current.strip())

    return chunks


def text_to_instruction_rows(chunks, source_name):
    """Convert text chunks into instruction/output training rows."""
    rows = []
    random.seed(42)

    for chunk in chunks:
        if not has_amharic(chunk):
            continue

        # Create continuation / completion examples from longer chunks
        # (No self-referencing "summarize" rows — those teach the model to echo)
        sentences = re.split(r"(?<=።)\s*", chunk)
        sentences = [s for s in sentences if s.strip()]

        if len(sentences) >= 4:
            split_point = len(sentences) // 2
            prefix = "። ".join(sentences[:split_point]) + "።"
            suffix = "። ".join(sentences[split_point:])
            if suffix and not suffix.endswith("።"):
                suffix += "።"

            template = random.choice(CONTINUATION_TEMPLATES)
            rows.append({
                "instruction": template,
                "input": prefix,
                "output": suffix,
                "source": source_name,
            })
        elif len(sentences) >= 2:
            # For shorter chunks, use first sentence as prompt, rest as output
            prefix = sentences[0] + ("።" if not sentences[0].endswith("።") else "")
            suffix = "። ".join(sentences[1:])
            if suffix and not suffix.endswith("።"):
                suffix += "።"

            template = random.choice(CONTINUATION_TEMPLATES)
            rows.append({
                "instruction": template,
                "input": prefix,
                "output": suffix,
                "source": source_name,
            })

    return rows


def fetch_wikipedia(max_rows):
    """Fetch Amharic Wikipedia articles."""
    from datasets import load_dataset

    print("Fetching Amharic Wikipedia...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.am", split="train")
    except Exception:
        try:
            ds = load_dataset("graelo/wikipedia", "20230601.am", split="train")
        except Exception as e:
            print(f"  Could not load Wikipedia: {e}")
            return []

    print(f"  Raw articles: {len(ds)}")

    all_chunks = []
    for item in ds:
        text = item.get("text", "")
        text = clean_text(text)
        if len(text) < 100:
            continue
        chunks = split_into_chunks(text)
        all_chunks.extend(chunks)
        if len(all_chunks) >= max_rows * 2:
            break

    rows = text_to_instruction_rows(all_chunks, "amharic_wikipedia")
    if max_rows and len(rows) > max_rows:
        random.shuffle(rows)
        rows = rows[:max_rows]

    print(f"  Generated {len(rows)} training rows from Wikipedia")
    return rows


def fetch_cc100(max_rows):
    """Fetch Amharic web text from allenai/c4 (replaces CC-100)."""
    from datasets import load_dataset

    print("Fetching C4 Amharic web text...")
    try:
        ds = load_dataset("allenai/c4", "am", split="train", streaming=True)
    except Exception as e:
        print(f"  Could not load CC-100: {e}")
        return []

    all_chunks = []
    count = 0
    try:
        for item in ds:
            text = item.get("text", "")
            if len(text) < 80 or not has_amharic(text):
                continue
            all_chunks.append(text.strip())
            count += 1
            if count >= max_rows * 3:
                break
            if count % 10000 == 0:
                print(f"  Processed {count} items...")
    except Exception as e:
        print(f"  Error while iterating C4: {e}. Will proceed with what was fetched so far.")

    rows = text_to_instruction_rows(all_chunks, "cc100_amharic")
    if max_rows and len(rows) > max_rows:
        random.shuffle(rows)
        rows = rows[:max_rows]

    print(f"  Generated {len(rows)} training rows from C4 Amharic")
    return rows


def fetch_masakhanews(max_rows):
    """Fetch MasakhaNEWS Amharic news articles."""
    from datasets import load_dataset

    print("Fetching MasakhaNEWS Amharic...")
    try:
        ds = load_dataset("masakhane/masakhanews", "amh", split="train")
    except Exception as e:
        print(f"  Could not load MasakhaNEWS: {e}")
        return []

    print(f"  Raw articles: {len(ds)}")

    rows = []
    for item in ds:
        text = item.get("text", "") or item.get("headline", "")
        headline = item.get("headline", "")
        category = item.get("label", "")

        if not text or len(text) < 50:
            continue

        # Summarization
        if headline and text != headline:
            rows.append({
                "instruction": "ይህን ዜና በአጭሩ ግለጽ።",
                "input": text[:800],
                "output": headline,
                "source": "masakhanews_amharic",
            })

        # Classification
        if category:
            label_map = {0: "ቢዝነስ", 1: "ኢንተርቴይንመንት", 2: "ጤና",
                         3: "ፖለቲካ", 4: "ስፖርት", 5: "ቴክኖሎጂ"}
            label = label_map.get(category, str(category))
            rows.append({
                "instruction": "የዚህን ዜና ምድብ ለይ።",
                "input": text[:500],
                "output": label,
                "source": "masakhanews_amharic",
            })

        if max_rows and len(rows) >= max_rows:
            break

    print(f"  Generated {len(rows)} training rows from MasakhaNEWS")
    return rows


def fetch_xlsum(max_rows):
    """Fetch XL-Sum Amharic summarization data."""
    from datasets import load_dataset

    print("Fetching XL-Sum Amharic...")
    try:
        ds = load_dataset("GEM/xlsum", "amharic", split="train")
    except Exception:
        try:
            ds = load_dataset("csebuetnlp/xlsum", "amharic", split="train")
        except Exception as e:
            print(f"  Could not load XL-Sum: {e}")
            return []

    print(f"  Raw articles: {len(ds)}")

    rows = []
    for item in ds:
        text = item.get("text", "")
        summary = item.get("summary", "")
        title = item.get("title", "")

        if not text or not summary:
            continue

        # Summarization pair
        rows.append({
            "instruction": "ይህን ጽሑፍ በአጭሩ ጻፍ።",
            "input": text[:1000],
            "output": summary,
            "source": "xlsum_amharic",
        })

        # Title generation
        if title:
            rows.append({
                "instruction": "ለዚህ ጽሑፍ ርዕስ ስጥ።",
                "input": text[:500],
                "output": title,
                "source": "xlsum_amharic",
            })

        if max_rows and len(rows) >= max_rows:
            break

    print(f"  Generated {len(rows)} training rows from XL-Sum")
    return rows


def fetch_amnli(max_rows):
    """Fetch AmNLI — Amharic Natural Language Inference."""
    from datasets import load_dataset

    print("Fetching AmNLI...")
    try:
        ds = load_dataset("masakhane/afrihate", "amh", split="train")
    except Exception:
        try:
            ds = load_dataset("Davlan/sib200", "amh_Ethi", split="train")
        except Exception as e:
            print(f"  Could not load AmNLI/SIB-200: {e}")
            return []

    print(f"  Raw items: {len(ds)}")

    rows = []
    for item in ds:
        # SIB-200 format: text + category
        text = item.get("text", "") or item.get("premise", "")
        label = item.get("category", "") or item.get("label", "")

        if not text:
            continue

        if isinstance(label, int):
            # NLI-style labels
            label_map = {0: "ትክክል (entailment)", 1: "ተቃራኒ (contradiction)",
                         2: "ግልጽ አይደለም (neutral)"}
            hypothesis = item.get("hypothesis", "")
            if hypothesis and label in label_map:
                rows.append({
                    "instruction": "ከመነሻ ዓረፍተ ነገሩ አንጻር ሁለተኛው ዓረፍተ ነገር ትክክል ነው፣ ተቃራኒ ነው፣ ወይስ ግልጽ አይደለም?",
                    "input": f"መነሻ: {text}\nግምት: {hypothesis}",
                    "output": label_map[label],
                    "source": "amharic_nli",
                })
        else:
            # Topic classification style
            rows.append({
                "instruction": "የዚህን ጽሑፍ ምድብ ለይ።",
                "input": text,
                "output": str(label),
                "source": "amharic_classification",
            })

        if max_rows and len(rows) >= max_rows:
            break

    print(f"  Generated {len(rows)} training rows")
    return rows


def fetch_walia_instructions(max_rows):
    """Fetch EthioNLP/walia-amharic-instructions."""
    from datasets import load_dataset
    
    print("Fetching Walia Amharic Instructions...")
    try:
        # Load the specific dataset that EthioNLP provides for instructions
        ds = load_dataset("EthioNLP/Amharic_Instruction_dataset", split="train")
    except Exception as e:
        print(f"  Could not load Walia Instructions: {e}")
        return []
        
    print(f"  Raw instructions: {len(ds)}")
    
    rows = []
    for item in ds:
        # Expected format: instruction, input, output
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        
        if not instruction or not output:
            continue
            
        rows.append({
            "instruction": instruction,
            "input": inp,
            "output": output,
            "source": "walia_instructions",
        })
        
        if max_rows and len(rows) >= max_rows:
            break
            
    print(f"  Generated {len(rows)} training rows from Walia Instructions")
    return rows


def fetch_israel_news(max_rows):
    """Fetch israel/Amharic-News-Text-classification-Dataset."""
    from datasets import load_dataset
    
    print("Fetching Israel News Classification...")
    try:
        ds = load_dataset("israel/Amharic-News-Text-classification-Dataset", split="train")
    except Exception as e:
        print(f"  Could not load Israel News: {e}")
        return []
        
    print(f"  Raw news articles: {len(ds)}")
    
    rows = []
    for item in ds:
        # Assuming typical news classification format
        # Check keys for actual field names (usually 'article' or 'text', and 'category' or 'label')
        text = item.get("article", "") or item.get("text", "") or item.get("content", "")
        label = item.get("category", "") or item.get("label", "")
        
        if not text or not has_amharic(text):
            continue
            
        text = clean_text(text)
        
        # Add summarization style (just use first 1000 chars as input, title as output if available)
        headline = item.get("headline", "") or item.get("title", "")
        if headline and text != headline:
            rows.append({
                "instruction": "ይህን ዜና በአጭሩ ግለጽ።",
                "input": text[:800],
                "output": headline,
                "source": "israel_news_amharic",
            })
            
        # Add classification style
        if label:
            rows.append({
                "instruction": "የዚህን ዜና ምድብ ለይ።",
                "input": text[:500],
                "output": str(label),
                "source": "israel_news_amharic",
            })
            
        # Add reading comprehension style for long text
        if len(text) > 200:
            chunks = split_into_chunks(text, max_chars=800)
            if chunks:
                chunk_rows = text_to_instruction_rows(chunks[:2], "israel_news_amharic")
                rows.extend(chunk_rows)
                
        if max_rows and len(rows) >= max_rows:
            break
            
    print(f"  Generated {len(rows)} training rows from Israel News")
    return rows


def fetch_rasyosef_sentences(max_rows):
    """Fetch rasyosef/amharic-sentences-corpus."""
    from datasets import load_dataset
    
    print("Fetching Rasyosef Amharic Sentences...")
    try:
        # Use streaming to avoid downloading all 6.4M sentences at once
        ds = load_dataset("rasyosef/amharic-sentences-corpus", split="train", streaming=True)
    except Exception as e:
        print(f"  Could not load Rasyosef Sentences: {e}")
        return []
        
    all_chunks = []
    count = 0
    current_chunk = ""
    
    for item in ds:
        text = item.get("text", "") or item.get("sentence", "")
        if len(text) < 10 or not has_amharic(text):
            continue
            
        # Combine short sentences into chunks for better context
        if len(current_chunk) + len(text) < 400:
            current_chunk = f"{current_chunk} {text}".strip()
        else:
            all_chunks.append(current_chunk)
            current_chunk = text
            count += 1
            
        if count >= max_rows * 2:  # Fetch extra to filter
            break
            
        if count % 10000 == 0:
            print(f"  Processed {count} chunks...")
            
    if current_chunk:
        all_chunks.append(current_chunk)
        
    rows = text_to_instruction_rows(all_chunks, "rasyosef_sentences")
    
    if max_rows and len(rows) > max_rows:
        random.shuffle(rows)
        rows = rows[:max_rows]
        
    print(f"  Generated {len(rows)} training rows from Rasyosef Sentences")
    return rows


SOURCES = {
    "wiki": fetch_wikipedia,
    "cc100": fetch_cc100,
    "news": fetch_masakhanews,
    "xlsum": fetch_xlsum,
    "amnli": fetch_amnli,
    "walia": fetch_walia_instructions,
    "israel_news": fetch_israel_news,
    "rasyosef": fetch_rasyosef_sentences,
}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch native Amharic text corpora for Tibeb training")
    parser.add_argument("--source", default="all",
                        choices=list(SOURCES.keys()) + ["all"],
                        help="Which source to fetch (default: all)")
    parser.add_argument("--max-rows", type=int, default=50000,
                        help="Max rows per source (default: 50000)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources_to_fetch = SOURCES if args.source == "all" else {args.source: SOURCES[args.source]}

    total = 0
    for name, fetcher in sources_to_fetch.items():
        print(f"\n{'='*50}")
        rows = fetcher(args.max_rows)
        if not rows:
            continue

        outfile = OUTPUT_DIR / f"{name}_amharic.jsonl"
        with open(outfile, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"  Saved {len(rows)} rows to {outfile}")
        total += len(rows)

    print(f"\n{'='*50}")
    print(f"Total rows across all sources: {total}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"\nNext: run merge_datasets.py to include these in training data.")


if __name__ == "__main__":
    main()
