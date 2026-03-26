"""
Tibeb AI Fine-Tuning Script
============================
Fine-tunes CohereForAI/aya-expanse-8b on the Tibeb unified training dataset.
Aya 8B is required — it's the only base model with real Amharic competency.

Usage:
    # Quick test (validates pipeline)
    python finetune_tibeb.py --test

    # Full SFT training
    python finetune_tibeb.py

    # Push trained adapter to HuggingFace
    python finetune_tibeb.py --push your-username/tibeb-model

Hardware:
    - Apple Silicon (MLX backend, auto-detected) — 24GB+ RAM
    - NVIDIA GPU (PyTorch + QLoRA backend) — 16GB+ VRAM
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

DATASET_PATH = "data/tibeb_unified_train.jsonl"
DPO_DATASET_PATH = "data/dpo/dpo_train.jsonl"
OUTPUT_DIR = "models/tibeb-sft"
DPO_OUTPUT_DIR = "models/tibeb-dpo"

TIBEB_SYSTEM_PROMPT = (
    "ቲብብ ነኝ — የኢትዮጵያ የፋይናንስ AI ረዳት።"
    " ስለ ኢንቨስትመንት፣ ቁጠባ፣ አክሲዮን ገበያ፣ ቲ-ቢል፣ እና የገንዘብ ጉዳዮች በአማርኛ እረዳለሁ።"
    " ሁልጊዜ አክብሮት ያለው፣ ትምህርታዊ፣ እና ግልጽ መልስ እሰጣለሁ።"
    " ቀጥተኛ የኢንቨስትመንት ምክር አልሰጥም — ተገቢውን ባለሙያ እንዲያማክሩ እመክራለሁ።"
)

MODELS = {
    "8b": "CohereForAI/aya-expanse-8b",
    "8b-4bit": "mlx-community/aya-expanse-8b-4bit",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}
DEFAULT_MODEL = "8b-4bit"


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


def format_row(row):
    """Convert a dataset row to chat format with Tibeb system prompt."""
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    output = row.get("output", "")

    if inp:
        user_msg = f"{instruction}\n\n{inp}"
    else:
        user_msg = instruction

    # Add system prompt for financial/Tibeb data; skip for generic NLP tasks
    source = row.get("source", "")
    financial_sources = {"tibeb_financial", "sujet_finance"}
    messages = []
    if source in financial_sources:
        messages.append({"role": "system", "content": TIBEB_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "content": output})

    return {"messages": messages}


def prepare_dataset(test=False, max_rows=None):
    """Load and format the unified JSONL dataset with quality filtering."""
    path = Path(DATASET_PATH)
    if not path.exists():
        print(f"Error: {DATASET_PATH} not found. Run scripts/merge_datasets.py first.")
        sys.exit(1)

    rows = []
    skipped = {"short": 0, "empty": 0, "echo": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("instruction") or not row.get("output"):
                skipped["empty"] += 1
                continue
            output = row["output"].strip()
            # Skip very short outputs — they cause nan loss with mask-prompt
            if len(output) < 10:
                skipped["short"] += 1
                continue
            # Skip self-referencing rows where output == input (echo data)
            inp = (row.get("input") or "").strip()
            if inp and output == inp:
                skipped["echo"] += 1
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

    if any(skipped.values()):
        print(f"  Filtered out: {skipped['empty']} empty, "
              f"{skipped['short']} short, {skipped['echo']} echo/self-ref")

    return rows


# ---------------------------------------------------------------------------
# MLX Backend (Apple Silicon)
# ---------------------------------------------------------------------------

def _write_yaml_config(config_dict, path):
    """Write a YAML config file for mlx-lm lora --config."""
    import yaml
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    return path


def train_mlx(model_id, test=False, resume=False, version="v2"):
    """Fine-tune using MLX LoRA on Apple Silicon."""
    from mlx_lm import load as mlx_load
    import mlx.core as mx

    ram_gb = get_ram_gb()

    print(f"\n{'='*60}")
    print(f"  MLX Fine-Tuning (Apple Silicon) — {version.upper()}")
    print(f"  Model: {model_id}")
    print(f"  RAM: {ram_gb:.0f}GB")
    print(f"  Backend: MLX")
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

    if version == "v3":
        output_dir = Path(OUTPUT_DIR) / "mlx-adapter-v3"
    else:
        output_dir = Path(OUTPUT_DIR) / "mlx-adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _get_mlx_config(model_id, ram_gb, test, version)

    if version == "v3" and configs.get("lr_schedule"):
        config_path = Path("configs")
        config_path.mkdir(exist_ok=True)
        yaml_config = {
            "model": model_id,
            "train": True,
            "data": str(data_dir),
            "adapter_path": str(output_dir),
            "fine_tune_type": "lora",
            "iters": configs["iters"],
            "batch_size": configs["batch_size"],
            "num_layers": configs["num_layers"],
            "learning_rate": configs["learning_rate"],
            "val_batches": 25,
            "steps_per_eval": configs["steps_per_eval"],
            "save_every": configs["save_every"],
            "max_seq_length": configs["seq_length"],
            "grad_checkpoint": True,
            "grad_accumulation_steps": configs.get("grad_accum", 1),
            "optimizer": configs.get("optimizer", "adam"),
            "seed": 42,
            "lora_parameters": configs["lora_parameters"],
            "lr_schedule": configs["lr_schedule"],
        }

        if resume:
            adapter_file = output_dir / "adapters.safetensors"
            if adapter_file.exists():
                yaml_config["resume_adapter_file"] = str(adapter_file)
                print(f"  Resuming from: {adapter_file}")

        yaml_path = _write_yaml_config(yaml_config, config_path / f"{version}_config.yaml")
        cmd_parts = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(yaml_path)]

        print(f"\n  Training config ({version.upper()}):")
        print(f"    Iterations:     {configs['iters']}")
        print(f"    Batch size:     {configs['batch_size']}")
        print(f"    Grad accum:     {configs.get('grad_accum', 1)} (eff. batch {configs['batch_size'] * configs.get('grad_accum', 1)})")
        print(f"    LoRA rank:      {configs['lora_parameters']['rank']}")
        print(f"    LoRA layers:    {configs['num_layers']} (-1 = all)")
        print(f"    Learning rate:  {configs['learning_rate']}")
        print(f"    LR schedule:    {configs['lr_schedule']['name']} + {configs['lr_schedule'].get('warmup', 0)} warmup")
        print(f"    Optimizer:      {configs.get('optimizer', 'adam')}")
        print(f"    Seq length:     {configs['seq_length']}")
        print(f"    Save every:     {configs['save_every']} steps")
    else:
        lora_args = {
            "model": model_id,
            "train": True,
            "data": str(data_dir),
            "adapter-path": str(output_dir),
            "iters": configs["iters"],
            "batch-size": configs["batch_size"],
            "num-layers": configs["num_layers"],
            "learning-rate": configs["learning_rate"],
            "val-batches": 25,
            "steps-per-eval": configs["steps_per_eval"],
            "save-every": configs["save_every"],
            "max-seq-length": configs["seq_length"],
            "grad-checkpoint": True,
            "seed": 42,
        }

        if resume:
            adapter_file = output_dir / "adapters.safetensors"
            if adapter_file.exists():
                lora_args["resume-adapter-file"] = str(adapter_file)
                print(f"  Resuming from: {adapter_file}")
            else:
                print(f"  Warning: No checkpoint found at {adapter_file}, starting fresh.")

        cmd_parts = [sys.executable, "-m", "mlx_lm", "lora"]
        for k, v in lora_args.items():
            if isinstance(v, bool):
                if v:
                    cmd_parts.append(f"--{k}")
            else:
                cmd_parts.extend([f"--{k}", str(v)])

        print(f"\n  Training config ({version.upper()}):")
        print(f"    Iterations:     {configs['iters']}")
        print(f"    Batch size:     {configs['batch_size']}")
        print(f"    LoRA layers:    {configs['num_layers']}")
        print(f"    Learning rate:  {configs['learning_rate']}")
        print(f"    Seq length:     {configs['seq_length']}")
        print(f"    Save every:     {configs['save_every']} steps")

    print(f"    Grad checkpoint: True")
    est_hours = configs["iters"] * 3 / 3600
    print(f"    Est. time:      {est_hours:.0f}-{est_hours*2:.0f} hours")
    print(f"\n  Running: {' '.join(cmd_parts)}\n")

    import subprocess
    result = subprocess.run(cmd_parts)
    if result.returncode != 0:
        print(f"\n  Error: mlx_lm.lora exited with code {result.returncode}")
        sys.exit(1)

    print(f"\n  Adapter saved to: {output_dir}")
    return output_dir


def _get_mlx_config(model_id, ram_gb, test, version):
    """Return training hyperparameters for the given version."""
    is_large = "8b" in model_id.lower() or "8B" in model_id
    is_quantized = "4bit" in model_id.lower()

    if test:
        return {
            "iters": 50, "steps_per_eval": 10, "save_every": 25,
            "batch_size": 1, "num_layers": 16, "seq_length": 512,
            "learning_rate": 1e-5, "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
            "lr_schedule": None,
        }

    if version == "v3":
        iters = 20000
        return {
            "iters": iters,
            "steps_per_eval": 500,
            "save_every": 1000,
            "batch_size": 1,
            "grad_accum": 4,
            "num_layers": -1,
            "seq_length": 512,
            "learning_rate": 1e-5,
            "optimizer": "adamw",
            "lora_parameters": {"rank": 16, "dropout": 0.05, "scale": 32.0},
            "lr_schedule": {
                "name": "cosine_decay",
                "arguments": [1e-5, iters, 1e-6],
                "warmup": 500,
                "warmup_init": 1e-7,
            },
        }

    # v2 default
    if is_large and not is_quantized and ram_gb <= 24:
        num_layers, seq_length, batch_size = 8, 256, 1
    elif is_large:
        num_layers, seq_length, batch_size = 16, 512, 1
    else:
        num_layers, seq_length, batch_size = 16, 512, 2

    return {
        "iters": 15000, "steps_per_eval": 500, "save_every": 1000,
        "batch_size": batch_size, "num_layers": num_layers, "seq_length": seq_length,
        "learning_rate": 1e-5, "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
        "lr_schedule": None,
    }


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
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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

# ---------------------------------------------------------------------------
# DPO Training (PyTorch — CUDA or MPS)
# ---------------------------------------------------------------------------

def prepare_dpo_dataset():
    """Load DPO preference pairs."""
    path = Path(DPO_DATASET_PATH)
    if not path.exists():
        print(f"Error: {DPO_DATASET_PATH} not found.")
        print(f"Run: python scripts/collect_preferences.py generate")
        print(f"     python scripts/collect_preferences.py rate")
        print(f"     python scripts/collect_preferences.py export")
        sys.exit(1)

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"  Loaded {len(rows)} DPO preference pairs")
    return rows


def train_dpo(model_id, test=False):
    """DPO training using the SFT adapter as the starting point."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset

    sft_adapter = Path(OUTPUT_DIR) / "pytorch-adapter"
    if not sft_adapter.exists():
        print(f"Error: SFT adapter not found at {sft_adapter}")
        print(f"Run SFT training first: python finetune_tibeb.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  DPO Training (Preference Alignment)")
    print(f"  Base: {model_id}")
    print(f"  SFT adapter: {sft_adapter}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    dpo_data = prepare_dpo_dataset()

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

    print("  Loading model with SFT adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(sft_adapter))
    model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    output_dir = Path(DPO_OUTPUT_DIR) / "pytorch-adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_idx = int(len(dpo_data) * 0.9)
    train_ds = Dataset.from_list(dpo_data[:split_idx])
    eval_ds = Dataset.from_list(dpo_data[split_idx:])

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1 if test else 3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        max_length=1024,
        max_prompt_length=512,
        beta=0.1,
        report_to="none",
        seed=42,
        max_steps=20 if test else -1,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\n  Starting DPO training...\n")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\n  DPO adapter saved to: {output_dir}")
    return output_dir


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
        print(f"  Run training first: python finetune_tibeb.py")
        sys.exit(1)

    api = HfApi()
    print(f"  Uploading to {repo_id}...")
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Done! View at: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tibeb AI Fine-Tuning")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run (200 rows, 50 steps)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL,
                        help=f"Model size (default: {DEFAULT_MODEL})")
    parser.add_argument("--dpo", action="store_true",
                        help="Run DPO training using preference data (requires SFT first)")
    parser.add_argument("--push", metavar="REPO_ID",
                        help="Push trained adapter to HuggingFace (e.g. nahommohan/tibeb-qwen-3b)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--v3", action="store_true",
                        help="Use v3 config (rank 16, all layers, cosine LR, adamw)")
    parser.add_argument("--backend", choices=["mlx", "cuda", "auto"], default="auto",
                        help="Training backend (default: auto-detect)")
    args = parser.parse_args()

    model_id = MODELS[args.model]
    version = "v3" if args.v3 else "v2"

    if args.push:
        push_model(args.push)
        return

    if args.dpo:
        backend = args.backend if args.backend != "auto" else detect_backend()
        if backend != "cuda":
            print("DPO training currently requires CUDA (NVIDIA GPU).")
            print("MLX DPO support is not yet available in mlx-lm.")
            sys.exit(1)
        train_dpo(model_id, test=args.test)
        return

    backend = args.backend if args.backend != "auto" else detect_backend()
    ram_gb = get_ram_gb()

    print(f"Hardware: {platform.machine()}, {ram_gb:.0f}GB RAM, backend={backend}")
    print(f"Model: {model_id} ({version})")

    if backend == "mlx":
        train_mlx(model_id, test=args.test, resume=args.resume, version=version)
    elif backend == "cuda":
        train_pytorch(model_id, test=args.test)
    elif backend == "mps":
        print("\n  Error: PyTorch MPS backend does not support QLoRA (4-bit) training.")
        print("  On Apple Silicon, install the MLX backend instead:")
        print("    pip install mlx mlx-lm")
        sys.exit(1)
    else:
        print("No suitable GPU backend found.")
        print("  Apple Silicon:  pip install mlx mlx-lm")
        print("  NVIDIA GPU:     pip install torch bitsandbytes")
        sys.exit(1)


if __name__ == "__main__":
    main()
