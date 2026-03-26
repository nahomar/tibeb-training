# Tibeb AI — Project Context

## What This Is
Fine-tuning pipeline for **Tibeb AI** — Ethiopia's Amharic financial assistant (part of EthioShare).
Competes with Neway (ESX official app) via Amharic-first, voice, and USSD access.

## Quick Reference

### Commands
```bash
source .venv/bin/activate          # always activate first
python3 scripts/fetch_amharic_corpus.py   # fetch native Amharic text
python3 scripts/merge_datasets.py         # merge all data sources
python3 finetune_tibeb.py --test          # quick SFT test
python3 finetune_tibeb.py                 # full SFT training
python3 finetune_tibeb.py --dpo           # DPO training (needs CUDA + SFT first)
python3 scripts/collect_preferences.py generate  # generate DPO candidates
python3 scripts/collect_preferences.py rate      # human rating CLI
python3 scripts/collect_preferences.py export    # export for DPO
python3 scripts/amharic_spelling.py --stats      # show spelling dict stats
```

### Environment
- Python: `python3` (not `python`)
- Venv: `.venv/` in project root
- Platform: Apple Silicon Mac (MLX backend auto-detected)
- Base model: `CohereForAI/aya-expanse-8b`

### Dataset (770K rows, 1.65 GB)
File: `data/tibeb_unified_train.jsonl`
- MT pairs: 200K | Sujet Finance (EN): 178K | Instructions: 122K
- Aya Collection: 100K | C4 Amharic: 50K | EthioSenti: 47K
- News: 41K | Wikipedia AM: 26K | ALFFA: 11K
- MasakhaNEWS: 1.3K | SIB-200: 701 | Synthetic QA: 120

### Spelling Normalizer (175 rules)
`scripts/amharic_spelling.py` — auto-applied during merge.
Key: ሠ→ሰ, ሃ/ሐ/ኀ→ሀ, ፀ→ጸ, plus financial/institution terms.

### HuggingFace Gotchas
- Broken (deprecated scripts): `cc100`, `mc4`, `csebuetnlp/xlsum`, `facebook/flores`
- Gated (need access): `uonlp/CulturaX`, `oscar-corpus/OSCAR-2301`
- Working: `allenai/c4` (am), `wikimedia/wikipedia`, `masakhane/masakhanews`, `Davlan/sib200`

### Key Architecture Decisions
- Aya 8B chosen for native Amharic competency (only viable option at this size)
- MLX LoRA for Apple Silicon, PyTorch QLoRA for NVIDIA
- DPO requires CUDA (mlx-lm doesn't support DPO yet)
- Spelling normalization runs on training data AND should run on inference outputs
- Spelling corrections cover ሰ/ሠ, ሀ/ሃ/ሐ/ኀ, ጸ/ፀ, plus financial terms
- System prompt added to financial training rows to teach Tibeb persona
- Echo/self-referencing data (output==input) filtered out during training
- Amharic grammar preserved: spelling normalizer no longer strips ው/ቱ articles

### Project Status — Production Pipeline
- [x] Data pipeline (fetch, merge, normalize, deduplicate)
- [x] Data quality: echo filter, system prompt, grammar-safe spelling
- [x] v3 training on 8B (val loss 1.133 at iter 2000, clean data)
- [x] 1,595 synthetic conversations generated (1,404 financial + 191 legacy)
- [ ] **Stage 0: Generate remaining synthetic conversations** (need API credits)
- [ ] **Stage 1: Continued Pretraining** on 400M tokens Amharic (72B model, 4x A100)
- [ ] **Stage 2: SFT** with curriculum learning (72B model)
- [ ] **Stage 3: DPO** alignment (5K-10K preference pairs)
- [ ] **Stage 4: RLHF** (optional — reward model + PPO)
- [ ] **Stage 5: Evaluation** (benchmarks + human eval)
- [ ] **Stage 6: Deploy** (quantize, API, voice, USSD)
- See `docs/PRODUCTION_TRAINING_PLAN.md` for full details

### Infrastructure Needed
- GPU: 4x A100 80GB (Lambda Labs ~$10/hr or RunPod ~$7/hr)
- Base model: Qwen/Qwen2.5-72B-Instruct (primary) or Aya-23-35B (backup)
- API credits: ~$500 for synthetic data generation
- Human annotators: 5-10 Amharic speakers for DPO/RLHF (~$2K-5K)
- Total budget: ~$3,500-7,500

### File Structure
```
finetune_tibeb.py                    # Main training (SFT + DPO)
scripts/
  fetch_amharic_corpus.py            # Fetch native Amharic from HF
  merge_datasets.py                  # Merge all sources → unified JSONL
  amharic_spelling.py                # 175-rule spelling normalizer
  generate_synthetic_data.py         # Claude API synthetic financial QA
  collect_preferences.py             # DPO preference collection
  prepare_hf_upload.py               # Push to HuggingFace Hub
data/
  tibeb_unified_train.jsonl          # Merged dataset (770K rows)
  native_amharic/                    # Fetched corpora (wiki, c4, news, sib200)
  synthetic_qa/raw_generated.json    # Generated financial conversations
  spelling_guide.md                  # Authoritative spelling reference
docs/
  RESEARCH_STRATEGY.md               # Competitive strategy vs Neway
models/
  tibeb-sft/                         # Trained adapters
```
