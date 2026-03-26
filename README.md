# Tibeb Training

Fine-tuning pipeline for **Tibeb AI** — Ethiopia's Amharic financial assistant.

## Overview

Tibeb is an Amharic-language AI assistant focused on financial literacy and education in the Ethiopian context. This repository contains the complete training pipeline: data collection, synthetic data generation, preprocessing, merging, spelling normalization, and fine-tuning using LoRA/QLoRA.

**Large files (training data, models)** are hosted on HuggingFace:
- Dataset: [nahomar/tibeb-training-data](https://huggingface.co/datasets/nahomar/tibeb-training-data)
- Model: [nahomar/tibeb-sft-adapter](https://huggingface.co/nahomar/tibeb-sft-adapter)

## Hardware Requirements

- **Apple Silicon Mac** (recommended): Uses MLX backend. 16GB+ RAM for 3B model, 32GB+ for 8B.
- **NVIDIA GPU**: Uses PyTorch + QLoRA. 16GB+ VRAM recommended.
- **Production training (72B)**: 4x A100 80GB (Lambda Labs or RunPod)

## Setup

```bash
pip install -r requirements.txt
```

Apple Silicon (MLX backend):

```bash
pip install mlx mlx-lm
```

NVIDIA GPU (QLoRA backend):

```bash
pip install torch bitsandbytes
```

## Data Pipeline

### 1. Fetch native Amharic corpora

Downloads Wikipedia, C4, MasakhaNEWS, and SIB-200 Amharic text from HuggingFace.

```bash
python scripts/fetch_amharic_corpus.py
```

### 2. Generate synthetic financial conversations (optional)

Requires an Anthropic API key. Generates Amharic financial Q&A conversations with proper honorifics across 150+ topics and 12 user profiles.

```bash
export ANTHROPIC_API_KEY=your_key
python scripts/gen_simple.py financial   # 1,800 financial conversations
python scripts/gen_simple.py general     # 600 general Amharic Q&A
python scripts/gen_simple.py voice       # 300 voice/USSD short-form
python scripts/gen_simple.py safety      # 150 safety & alignment
python scripts/gen_simple.py deep        # 90 multi-turn deep dives
```

### 3. Merge all datasets into a unified training file

Combines 10+ data sources, applies spelling normalization, deduplicates, and shuffles.

```bash
python scripts/merge_datasets.py
```

### 4. Fine-tune the model

```bash
# Quick test (validates the pipeline with 200 rows, 50 steps)
python finetune_tibeb.py --test

# Full SFT training v3 (rank 16, all layers, cosine LR, adamw)
python finetune_tibeb.py --v3

# Or use a YAML config directly
python -m mlx_lm lora --config configs/v4_config.yaml
```

### 5. DPO Alignment (requires CUDA GPU)

```bash
python scripts/collect_preferences.py generate   # generate DPO candidates
python scripts/collect_preferences.py rate        # human rating CLI
python scripts/collect_preferences.py export      # export for DPO training
python finetune_tibeb.py --dpo                    # run DPO training
```

### 6. Push to HuggingFace (optional)

```bash
python finetune_tibeb.py --push your-username/tibeb-model
python scripts/prepare_hf_upload.py --repo your-username/tibeb-training-data
```

## Data Sources

| Source | ~Rows | Language | Description |
|--------|-------|----------|-------------|
| EthioNLP Instructions | 122K | Amharic | Instruction-following tasks |
| Amharic MT | 200K | Am/En | Translation pairs (filtered for Amharic output) |
| Amharic News | 41K | Mixed | News classification |
| Aya Collection | 100K | Amharic | Diverse NLP tasks |
| EthioSenti | 47K | Amharic | Sentiment analysis |
| ALFFA Transcriptions | 10K+ | Amharic | Voice transcription text |
| Native Amharic (Wiki, C4, etc.) | 78K | Amharic | 4 corpus sources |
| **Tibeb Synthetic Financial** | **1,595** | **Amharic** | **Generated financial conversations (5x upsampled)** |

**Merged dataset:** ~692K rows after dedup and filtering.

## Project Structure

```
├── finetune_tibeb.py              # Main fine-tuning script (MLX + PyTorch)
├── requirements.txt               # Python dependencies
├── CLAUDE.md                      # AI assistant project context
├── configs/
│   ├── v3_config.yaml             # v3 training config (rank 16, cosine LR)
│   └── v4_config.yaml             # v4 training config
├── docs/
│   ├── PRODUCTION_TRAINING_PLAN.md  # 6-stage production pipeline
│   └── RESEARCH_STRATEGY.md        # Competitive strategy vs Neway
├── notebooks/
│   ├── tibeb_train_colab.ipynb    # Google Colab training notebook
│   └── tibeb_colab_v2.ipynb       # Colab v2 notebook
├── data/
│   ├── spelling_guide.md          # Authoritative Amharic spelling reference
│   ├── useful_resources.txt       # Reference links
│   ├── alffa_transcriptions.json  # ALFFA voice transcriptions
│   ├── synthetic_qa/              # Generated conversations (1,595 convos)
│   ├── dpo/                       # DPO preference data
│   ├── human_authored/            # Human-authored training examples + form
│   ├── *.jsonl                    # Large source datasets (on HuggingFace)
│   └── tibeb_unified_train.jsonl  # Merged dataset (on HuggingFace)
├── scripts/
│   ├── generate_synthetic_data.py # Original synthetic data generator
│   ├── gen_simple.py              # Production synthetic data generator
│   ├── merge_datasets.py          # Merge, normalize, deduplicate, shuffle
│   ├── amharic_spelling.py        # 175-rule spelling normalizer
│   ├── fetch_amharic_corpus.py    # Download native Amharic corpora
│   ├── eval_model.py              # Model evaluation script
│   ├── collect_preferences.py     # DPO preference collection
│   └── prepare_hf_upload.py       # Upload dataset to HuggingFace Hub
├── models/                        # Trained adapters (on HuggingFace)
└── logs/                          # Training logs (local only)
```

## Spelling Normalization

Tibeb enforces authoritative Amharic spellings (see `data/spelling_guide.md`). The normalizer in `scripts/amharic_spelling.py` (175 rules) is applied automatically during dataset merging. Key rules:

- ሰላም not ሠላም, ሰው not ሠው, ሀገር not ሃገር
- እርስዎ not እርሶ (formal address)
- ቲ-ቢል with hyphen, ብር not ብርር, ምንዛሬ not ምንዛሪ
- Arabic numerals for financial data, Ethiopic punctuation (።  ፣)

## Base Models

| Size | Model | Recommended For |
|------|-------|-----------------|
| 72B | Qwen/Qwen2.5-72B-Instruct | Production (4x A100 80GB) |
| 8B | CohereForAI/aya-expanse-8b | CUDA GPU or 32GB+ Apple Silicon |
| 3B | Qwen/Qwen2.5-3B-Instruct | 16GB+ Apple Silicon |
| 1.5B | Qwen/Qwen2.5-1.5B-Instruct | 8GB+ Apple Silicon |

## Production Training Pipeline

See [`docs/PRODUCTION_TRAINING_PLAN.md`](docs/PRODUCTION_TRAINING_PLAN.md) for the full 6-stage pipeline:

1. **Data Factory** — Generate 50K conversations, DPO pairs, RLHF annotations
2. **Continued Pretraining** — 400M tokens Amharic on 72B model
3. **SFT** — Curriculum learning (general → Amharic → finance → persona)
4. **DPO** — 5K-10K human preference pairs
5. **RLHF** — Reward model + PPO (optional)
6. **Deploy** — Quantize, API, voice, USSD

**Budget:** ~$3,500-7,500 | **Timeline:** 5-6 weeks

## Resources

- [EthioNLP on HuggingFace](https://huggingface.co/EthioNLP)
- [Aya Model Paper (arXiv:2402.08015)](https://arxiv.org/abs/2402.08015)
- [Production Training Plan](docs/PRODUCTION_TRAINING_PLAN.md)
- [Research Strategy](docs/RESEARCH_STRATEGY.md)
