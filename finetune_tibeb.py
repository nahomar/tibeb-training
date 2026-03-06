"""
Tibeb AI Fine-Tuning Script
============================
Fine-tunes a language model on the Tibeb unified training dataset.

Usage:
    # Quick test (validates pipeline)
    python finetune_tibeb.py --stage 1 --test

    # Full SFT training
    python finetune_tibeb.py --stage both

    # Push trained adapter to HuggingFace
    python finetune_tibeb.py --push nahommohan/tibeb-aya-8b

Hardware:
    - Apple Silicon (MLX backend, auto-detected)
    - NVIDIA GPU (PyTorch + QLoRA backend)
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

DATASET_PATH = "data/tibeb_unified_train.jsonl"
OUTPUT_DIR = "models/tibeb-sft"

# Models sized for available hardware
MODELS = {
    "8b": "CohereForAI/aya-expanse-8b",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def detect_backend():
    machine = platform.machine()
    if machine == "arm64" and platform.system() == "Darwin":
        try:
            import mlx.core
            return "mlx"
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    if platform.system() == "Darwin":
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        return int(out) / (1024 ** 3)
    return 8


def recommend_model(ram_gb, backend):
    if backend == "cuda":
        return "8b"
    if ram_gb >= 32:
        return "8b"
    if ram_gb >= 16:
        return "3b"
    return "1.5b"


def format_row(row):
    """Convert a dataset row to chat format."""
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    output = row.get("output", "")

    if inp:
        user_msg = f"{instruction}\n\n{inp}"
    else:
        user_msg = instruction

    return {"messages": [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]}


def prepare_dataset(test=False, max_rows=None):
    """Load and format the unified JSONL dataset."""
    path = Path(DATASET_PATH)
    if not path.exists():
        print(f"Error: {DATASET_PATH} not found. Run scripts/merge_datasets.py first.")
        sys.exit(1)

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("instruction") or not row.get("output"):
                continue
            rows.append(format_row(row))

    if test:
        rows = rows[:200]
        print(f"  Test mode: using {len(rows)} rows")
    elif max_rows:
        rows = rows[:max_rows]
        print(f"  Using {len(rows)} rows (capped)")
    else:
        print(f"  Loaded {len(rows)} rows")

    return rows


# ---------------------------------------------------------------------------
# MLX Backend (Apple Silicon)
# ---------------------------------------------------------------------------

def train_mlx(model_id, test=False):
    """Fine-tune using MLX on Apple Silicon."""
    from mlx_lm import load as mlx_load
    import mlx.core as mx

    print(f"\n{'='*60}")
    print(f"  MLX Fine-Tuning (Apple Silicon)")
    print(f"  Model: {model_id}")
    print(f"  Backend: MLX {mx.__version__}")
    print(f"{'='*60}\n")

    train_data = prepare_dataset(test=test)

    data_dir = Path("data/mlx_train")
    data_dir.mkdir(parents=True, exist_ok=True)

    split_idx = int(len(train_data) * 0.95)
    with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train_data[:split_idx]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(data_dir / "valid.jsonl", "w", encoding="utf-8") as f:
        for row in train_data[split_idx:]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Train: {split_idx}, Valid: {len(train_data) - split_idx}")

    output_dir = Path(OUTPUT_DIR) / "mlx-adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    iters = 50 if test else 1000
    lora_args = {
        "model": model_id,
        "train": True,
        "data": str(data_dir),
        "adapter-path": str(output_dir),
        "iters": iters,
        "batch-size": 1,
        "num-layers": 8,
        "learning-rate": 1e-5,
        "val-batches": 5,
        "steps-per-eval": 100 if not test else 10,
        "max-seq-length": 512,
        "mask-prompt": True,
        "seed": 42,
    }

    cmd_parts = [sys.executable, "-m", "mlx_lm", "lora"]
    for k, v in lora_args.items():
        if isinstance(v, bool):
            if v:
                cmd_parts.append(f"--{k}")
        else:
            cmd_parts.extend([f"--{k}", str(v)])

    print(f"\n  Running: {' '.join(cmd_parts)}\n")
    import subprocess
    result = subprocess.run(cmd_parts)
    if result.returncode != 0:
        print(f"\n  Error: mlx_lm.lora exited with code {result.returncode}")
        sys.exit(1)

    print(f"\n  Adapter saved to: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# PyTorch Backend (CUDA GPU)
# ---------------------------------------------------------------------------

def train_pytorch(model_id, test=False):
    """Fine-tune using PyTorch + QLoRA on NVIDIA GPU."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    print(f"\n{'='*60}")
    print(f"  PyTorch QLoRA Fine-Tuning")
    print(f"  Model: {model_id}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    train_data = prepare_dataset(test=test)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_chat(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    output_dir = Path(OUTPUT_DIR) / "pytorch-adapter"

    num_epochs = 1 if test else 3
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2 if not test else 1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        max_grad_norm=1.0,
        report_to="wandb" if not test else "none",
        run_name="tibeb-sft" if not test else "tibeb-sft-test",
        seed=42,
        max_steps=50 if test else -1,
    )

    split_idx = int(len(train_data) * 0.95)
    from datasets import Dataset
    train_ds = Dataset.from_list(train_data[:split_idx])
    eval_ds = Dataset.from_list(train_data[split_idx:])

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=format_chat,
        max_seq_length=512,
        packing=True,
    )

    print("\n  Starting training...\n")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\n  Adapter saved to: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Push to HuggingFace
# ---------------------------------------------------------------------------

def push_model(repo_id):
    """Push the trained adapter to HuggingFace Hub."""
    from huggingface_hub import HfApi

    mlx_path = Path(OUTPUT_DIR) / "mlx-adapter"
    pytorch_path = Path(OUTPUT_DIR) / "pytorch-adapter"

    if mlx_path.exists():
        adapter_path = mlx_path
        print(f"  Found MLX adapter at {adapter_path}")
    elif pytorch_path.exists():
        adapter_path = pytorch_path
        print(f"  Found PyTorch adapter at {adapter_path}")
    else:
        print(f"  Error: No trained adapter found in {OUTPUT_DIR}/")
        print(f"  Run training first: python finetune_tibeb.py --stage both")
        sys.exit(1)

    api = HfApi()
    print(f"  Uploading to {repo_id}...")
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Done! View at: https://huggingface.co/models/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tibeb AI Fine-Tuning")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                        help="Training stage: 1=SFT, 2=DPO, both=SFT+DPO")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run (200 rows, 50 steps)")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Model size (auto-detected if not set)")
    parser.add_argument("--push", metavar="REPO_ID",
                        help="Push trained adapter to HuggingFace (e.g. nahommohan/tibeb-aya-8b)")
    parser.add_argument("--backend", choices=["mlx", "cuda", "auto"], default="auto",
                        help="Training backend (default: auto-detect)")
    args = parser.parse_args()

    if args.push:
        push_model(args.push)
        return

    backend = args.backend if args.backend != "auto" else detect_backend()
    ram_gb = get_ram_gb()

    print(f"Hardware: {platform.machine()}, {ram_gb:.0f}GB RAM, backend={backend}")

    if not args.model:
        size = recommend_model(ram_gb, backend)
        print(f"Auto-selected model: {size} ({MODELS[size]})")
        if size != "8b":
            print(f"  Note: 8B model needs ≥16GB RAM on Mac or a CUDA GPU.")
            print(f"  Override with: --model 8b")
    else:
        size = args.model

    model_id = MODELS[size]

    if backend == "mlx":
        train_mlx(model_id, test=args.test)
    elif backend in ("cuda", "mps", "cpu"):
        if backend != "cuda":
            print(f"\n  Warning: {backend} backend is slow for training.")
            print(f"  Consider using a cloud GPU or MLX on Apple Silicon.\n")
        train_pytorch(model_id, test=args.test)
    else:
        print(f"No suitable backend found. Install mlx or torch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
