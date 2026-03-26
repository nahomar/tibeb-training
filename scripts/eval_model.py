"""
Evaluate a trained Tibeb adapter on Amharic financial prompts.

Usage:
    python scripts/eval_model.py                          # use latest adapter
    python scripts/eval_model.py --adapter models/tibeb-sft/mlx-adapter
    python scripts/eval_model.py --checkpoint 12000       # specific checkpoint
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

EVAL_PROMPTS = [
    ("basic", "ቲ-ቢል ምንድነው?"),
    ("advice", "ለጀማሪ ኢንቨስተር ምን ትመክራለህ?"),
    ("comparison", "ቁጠባ ባንክ ውስጥ ማስቀመጥ ወይስ ኢንቨስት ማድረግ ይሻላል?"),
    ("esx", "የኢትዮጵያ ሴኩሪቲስ ኤክስቼንጅ ምንድነው?"),
    ("inflation", "የኢንፍሌሽን ተጽዕኖ ከቁጠባ ላይ ምንድነው?"),
    ("telebirr", "ቴሌብር ተጠቅሜ አክሲዮን መግዛት እችላለሁ?"),
    ("risk", "የአክሲዮን ገበያ ሪስክ ምንድነው?"),
    ("greeting", "ሰላም! ስለ ኢትዮጵያ የካፒታል ገበያ ንገረኝ።"),
]


def load_model(adapter_path):
    from mlx_lm import load

    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_id = config.get("model", "mlx-community/aya-expanse-8b-4bit")
    else:
        model_id = "mlx-community/aya-expanse-8b-4bit"

    print(f"  Model: {model_id}")
    print(f"  Adapter: {adapter_path}")
    model, tokenizer = load(model_id, adapter_path=adapter_path)
    return model, tokenizer


def run_eval(model, tokenizer, max_tokens=300, temp=0.5):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temp)

    results = []
    print(f"\n{'='*70}")
    print(f"  Tibeb Model Evaluation")
    print(f"  {len(EVAL_PROMPTS)} prompts | max_tokens={max_tokens} | temp={temp}")
    print(f"{'='*70}")

    for tag, prompt in EVAL_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.time()
        response = generate(model, tokenizer, prompt=prompt_text,
                            max_tokens=max_tokens, sampler=sampler)
        elapsed = time.time() - start

        results.append({
            "tag": tag,
            "prompt": prompt,
            "response": response,
            "time_s": round(elapsed, 1),
        })

        print(f"\n{'─'*70}")
        print(f"  [{tag}] {prompt}")
        print(f"  ({elapsed:.1f}s)")
        print(f"{'─'*70}")
        for line in response.strip().split("\n"):
            print(f"  {line}")

    return results


def setup_checkpoint_adapter(adapter_dir, checkpoint):
    """Create a temp adapter dir with a specific checkpoint's weights."""
    adapter_dir = Path(adapter_dir)
    ckpt_file = adapter_dir / f"{checkpoint:07d}_adapters.safetensors"
    if not ckpt_file.exists():
        print(f"Error: Checkpoint {ckpt_file} not found.")
        available = sorted(adapter_dir.glob("*_adapters.safetensors"))
        print(f"Available: {[f.name for f in available]}")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="tibeb_eval_")
    shutil.copy2(adapter_dir / "adapter_config.json", tmp_dir)
    shutil.copy2(ckpt_file, Path(tmp_dir) / "adapters.safetensors")
    return tmp_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tibeb model")
    parser.add_argument("--adapter", type=str, default="models/tibeb-sft/mlx-adapter",
                        help="Path to adapter directory")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Use specific checkpoint iteration (e.g. 12000)")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    tmp_dir = None
    adapter_path = args.adapter

    if args.checkpoint:
        tmp_dir = setup_checkpoint_adapter(args.adapter, args.checkpoint)
        if not tmp_dir:
            return
        adapter_path = tmp_dir
        print(f"  Using checkpoint: iter {args.checkpoint}")

    try:
        model, tokenizer = load_model(adapter_path)
        results = run_eval(model, tokenizer, args.max_tokens, args.temp)

        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n  Results saved to {args.save}")

        print(f"\n{'='*70}")
        print(f"  Evaluation complete: {len(results)} prompts")
        avg_time = sum(r["time_s"] for r in results) / len(results)
        print(f"  Average response time: {avg_time:.1f}s")
        print(f"{'='*70}")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
