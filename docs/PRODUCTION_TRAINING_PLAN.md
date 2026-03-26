# Tibeb AI — Production Training Plan

**Goal:** Train a state-of-the-art Amharic financial AI that rivals frontier models in its domain.

**Approach:** Follow the same multi-stage pipeline used by Anthropic (Claude), OpenAI (GPT), and DeepSeek — adapted for a domain-specific Amharic model.

---

## Architecture Overview

```
Stage 0: Data Factory          → 50K+ synthetic conversations
Stage 1: Continued Pretraining → Teach the base model deep Amharic fluency
Stage 2: Supervised Fine-Tuning → Teach instruction following + Tibeb persona
Stage 3: DPO Alignment         → Align with human preferences
Stage 4: RLHF (Optional)       → Reward model + PPO for final polish
Stage 5: Evaluation & Safety   → Benchmark, human eval, red-teaming
Stage 6: Deployment             → Quantize, serve, integrate voice/USSD
```

---

## Stage 0: Data Factory (2-3 weeks)

This is the most important stage. Model quality = data quality. Frontier labs spend 80% of effort here.

### 0.1 Synthetic Conversation Generation (50K target)

| Category | Count | Description |
|----------|-------|-------------|
| Financial core | 15,000 | T-Bills, ESX, stocks, bonds, savings, insurance, budgeting |
| Financial edge cases | 5,000 | Scam detection, bad advice refusal, "I don't know" responses |
| General Amharic Q&A | 10,000 | Culture, history, daily life, education, health, technology |
| Voice-style short form | 5,000 | USSD menu interactions, quick Q&A (1-2 sentence responses) |
| Multi-turn deep dives | 5,000 | 5-10 turn conversations on complex financial topics |
| Safety & alignment | 3,000 | Harmful request refusal, bias handling, cultural sensitivity |
| Persona consistency | 2,000 | Tibeb identity, honorifics, formal/informal switching |
| Code-switching | 2,000 | Amharic-English mixed queries (common in Ethiopian tech/finance) |
| Ethiopian regulations | 3,000 | NBE rules, ECMA regulations, tax law, business registration |

**Generation strategy:**
- Use Claude Sonnet for quality (not Haiku — Amharic quality matters)
- Cost estimate: 50K × ~$0.01/conversation = **~$500**
- Parallelize with 5-8 concurrent requests
- Auto-validate: check Amharic ratio, response length, JSON validity
- Human spot-check 5% sample (~2,500 conversations)

**Topic expansion needed:**
- Ethiopian tax brackets and filing
- Microfinance and edir/equb (traditional savings)
- Agricultural commodities (coffee, teff, sesame)
- Real estate investment in Ethiopia
- Diaspora remittance and investment
- Islamic finance (Sharia-compliant products)
- Student financial literacy
- Small business and startup financing
- Ethiopian pension system (public + private)
- Government bonds beyond T-Bills

### 0.2 Curated High-Quality Data

Beyond synthetic, collect:

| Source | Estimated Size | How to Get |
|--------|---------------|------------|
| NBE publications (translated) | 500 docs | Scrape nbe.gov.et, translate key docs |
| ESX investor guides | 50 docs | ESX website, translate to Amharic instruction pairs |
| Ethiopian financial news (Amharic) | 10K articles | Fana, EBC, Reporter Amharic |
| Equb/Edir explainers | 200 conversations | Generate with cultural context |
| Ethiopian proverbs about money | 500 pairs | Curate + explain each one |
| Telebirr/CBE user guides | 100 docs | Convert to Q&A format |

### 0.3 DPO Preference Data (5K-10K pairs)

For each prompt, generate 3 responses at different quality levels:
- **Chosen:** Correct Amharic, accurate info, proper honorifics, educational tone
- **Rejected:** Mix of failure modes:
  - English leakage (responds in English)
  - Wrong honorifics (uses አንተ for elders)
  - Gives direct investment advice (should refuse)
  - Inaccurate financial info
  - Robotic/unnatural Amharic
  - Too short/unhelpful

**Method:** Generate chosen/rejected pairs programmatically with Claude, then have 3-5 native Amharic speakers validate and rate.

### 0.4 RLHF Annotation Data (2K-5K)

Hire 5-10 native Amharic speakers (preferably with finance background) to:
- Rate model outputs on 5 dimensions: fluency, accuracy, helpfulness, safety, cultural appropriateness
- Annotate ~500-1000 examples each
- Use Argilla or Label Studio for annotation interface
- Budget: ~$2,000-5,000 for annotation

---

## Stage 1: Continued Pretraining (CPT)

**Goal:** Make the base model deeply fluent in Amharic before fine-tuning.

### Why CPT matters:
Frontier models like Claude and GPT train on trillions of tokens. A 70B model has seen some Amharic, but not enough for native-level fluency. CPT bridges this gap.

### Data for CPT:
| Source | Size | Description |
|--------|------|-------------|
| Amharic Wikipedia | ~50M tokens | Clean encyclopedic text |
| C4 Amharic | ~200M tokens | Web-crawled Amharic |
| Amharic news corpus | ~100M tokens | News articles |
| Amharic books/literature | ~50M tokens | If available |
| Ethiopian gov publications | ~20M tokens | Official documents |
| **Total** | **~400M tokens** | |

### CPT Config:
```yaml
model: Qwen/Qwen2.5-72B-Instruct  # or meta-llama/Llama-3.1-70B-Instruct
method: full fine-tune (not LoRA — CPT needs full weight updates)
learning_rate: 2e-5
lr_scheduler: cosine with warmup
warmup_steps: 500
batch_size: 128 (via gradient accumulation)
max_seq_length: 4096
epochs: 2-3 over the Amharic corpus
precision: bf16
framework: DeepSpeed ZeRO-3 or FSDP
hardware: 4x A100 80GB or 2x H100 80GB
estimated_time: 48-72 hours
```

### Important CPT considerations:
- Use a **low learning rate** (2e-5) to avoid catastrophic forgetting of English/general knowledge
- Mix in 10-20% English data to maintain bilingual capability
- Monitor perplexity on both Amharic and English validation sets
- Save checkpoints every 1000 steps

---

## Stage 2: Supervised Fine-Tuning (SFT)

**Goal:** Teach the model to follow instructions, adopt the Tibeb persona, and respond correctly to financial queries.

### Base for SFT:
Use the CPT checkpoint from Stage 1.

### SFT Data Curriculum:
Train in order of increasing specificity (curriculum learning):

| Phase | Data | Rows | Epochs |
|-------|------|------|--------|
| 2a: General instruction following | Aya Collection + EthioNLP instructions | 222K | 1 |
| 2b: Amharic fluency | Native Amharic corpus (continuations) | 300K | 1 |
| 2c: Financial knowledge | Sujet Finance (EN) + financial conversations | 200K | 2 |
| 2d: Tibeb persona | 50K synthetic + curated financial data | 50K | 3 |

**Key insight from frontier labs:** The last phase (persona/domain data) should be trained for more epochs with a lower learning rate. This is like the "fine-tuning on fine-tuning" approach Claude and GPT use.

### SFT Config:
```yaml
method: QLoRA (rank 128, alpha 256) or full fine-tune if budget allows
learning_rate: 1e-5 → decay to 1e-6
lr_scheduler: cosine
warmup_steps: 200
batch_size: 64 (via gradient accumulation)
max_seq_length: 2048  # Much longer than current 512!
lora_target_modules: all linear layers
mask_prompt: true  # Only train on assistant responses
precision: bf16
framework: TRL SFTTrainer + DeepSpeed
hardware: 4x A100 80GB
estimated_time: 24-48 hours
```

### System prompt (baked into training):
```
ቲብብ ነኝ — የኢትዮጵያ የፋይናንስ AI ረዳት። ስለ ኢንቨስትመንት፣ ቁጠባ፣ አክሲዮን ገበያ፣
ቲ-ቢል፣ እና የገንዘብ ጉዳዮች በአማርኛ እረዳለሁ። ሁልጊዜ አክብሮት ያለው፣ ትምህርታዊ፣
እና ግልጽ መልስ እሰጣለሁ። ቀጥተኛ የኢንቨስትመንት ምክር አልሰጥም — ተገቢውን ባለሙያ
እንዲያማክሩ እመክራለሁ።
```

---

## Stage 3: DPO Alignment

**Goal:** Align the model's outputs with human preferences — prefer helpful, accurate, safe Amharic responses.

### DPO Data:
- 5K-10K preference pairs (from Stage 0.3)
- Each pair: prompt + chosen response + rejected response
- Focus areas:
  - Amharic fluency (prefer native-sounding over translated)
  - Honorific correctness (prefer correct እርስዎ/አንተ/አንቺ usage)
  - Financial accuracy (prefer accurate over vague)
  - Safety (prefer refusal over harmful advice)

### DPO Config:
```yaml
method: DPO
beta: 0.1
learning_rate: 5e-7  # Very low — DPO is sensitive
lr_scheduler: cosine
batch_size: 16
max_length: 2048
max_prompt_length: 1024
epochs: 1-3
framework: TRL DPOTrainer + DeepSpeed
hardware: 4x A100 80GB
estimated_time: 6-12 hours
```

### DPO tips from frontier labs:
- Start from the best SFT checkpoint (not the last one)
- Use a very low learning rate — DPO can degrade model quality if too aggressive
- Monitor reward accuracy (should be >70% on held-out pairs)
- If quality degrades, reduce beta or learning rate

---

## Stage 4: RLHF (Optional — Maximum Quality)

**Goal:** Train a reward model, then use PPO to further align the base model.

### 4.1 Reward Model
- Take the DPO model, add a reward head
- Train on the same preference data (5K-10K pairs)
- Validate: reward model should agree with humans >75% of the time

### 4.2 PPO Training
- Use the reward model to score outputs
- PPO optimizes the policy (Tibeb) to maximize reward
- Very compute-intensive — ~2x the cost of SFT
- Requires careful KL penalty tuning to avoid reward hacking

### Budget:
- Reward model training: ~6 hours on 4x A100
- PPO training: ~24-48 hours on 4x A100
- Total: ~$500-1,000 in GPU cost

### Whether to do RLHF:
- DPO alone gets you 80% of the way there
- RLHF adds the final polish but is complex and expensive
- Recommendation: Start with DPO, add RLHF only if needed after evaluation

---

## Stage 5: Evaluation & Safety

### Automated Benchmarks:
| Benchmark | What it Tests | How |
|-----------|---------------|-----|
| Amharic perplexity | Fluency | Measure on held-out Amharic text |
| Financial accuracy | Domain knowledge | 200 curated financial questions + gold answers |
| Honorific accuracy | Cultural correctness | 100 prompts with known correct address form |
| Safety refusal rate | Alignment | 100 harmful prompts — should refuse all |
| English leakage rate | Language consistency | 200 Amharic prompts — measure % English in output |
| Response quality score | Helpfulness | LLM-as-judge (Claude rates Tibeb outputs 1-5) |

### Human Evaluation:
- 5 native Amharic speakers
- Rate 200 responses each on: fluency, accuracy, helpfulness, safety
- Compare against base model (before fine-tuning) and GPT-4 Amharic output
- Budget: ~$500-1,000

### Red-teaming:
- Test adversarial prompts (jailbreaks, prompt injection)
- Test financial misinformation resistance
- Test cultural sensitivity edge cases
- Test code-switching attacks (English prompt to bypass safety)

---

## Stage 6: Deployment

### Quantization:
```bash
# Quantize 72B model to 4-bit for efficient inference
python -m vllm.entrypoints.openai.api_server \
  --model tibeb-72b-v1 \
  --quantization awq \
  --max-model-len 4096
```

### Serving options:
| Option | Cost | Latency | Best For |
|--------|------|---------|----------|
| vLLM on 2x A100 | ~$5/hr | ~1-2s | Production API |
| TGI on 1x H100 | ~$4/hr | ~1-2s | Production API |
| llama.cpp on Mac | Free | ~5-10s | Development/demo |
| Groq/Together API | ~$0.001/req | <1s | Serverless |

### Integration:
1. **REST API** — FastAPI wrapper around vLLM
2. **Voice** — Addis AI Realtime API → Tibeb API → TTS
3. **USSD** — Ethio Telecom USSD gateway → Tibeb API (short responses)
4. **EthioShare app** — Embed via WebView or native SDK

---

## Base Model Selection

### Recommendation: Qwen 2.5-72B-Instruct

| Model | Params | Amharic | Finance | License | Notes |
|-------|--------|---------|---------|---------|-------|
| **Qwen 2.5-72B-Instruct** | 72B | Good | Good | Apache 2.0 | Best overall for multilingual + instruction following |
| Llama 3.1-70B-Instruct | 70B | OK | Good | Llama license | Strong general, weaker Amharic |
| Aya-23-35B | 35B | Best | OK | CC-BY-NC | Best native Amharic but smaller |
| DeepSeek-V2.5-236B (MoE) | 236B (21B active) | OK | Excellent | MIT | Huge but efficient MoE, great for finance |

**Primary choice:** Qwen 2.5-72B — best balance of size, multilingual support, and open license.

**Backup:** If budget is tight, use Aya-23-35B (half the GPU cost, best Amharic out-of-box).

**Long-term:** DeepSeek MoE architecture — very efficient for inference at massive scale.

---

## GPU Infrastructure

### Recommended Providers:

| Provider | 4x A100 80GB | 2x H100 80GB | Notes |
|----------|-------------|-------------|-------|
| Lambda Labs | ~$10/hr | ~$13/hr | Best UX, pre-installed frameworks |
| RunPod | ~$7/hr | ~$10/hr | Cheaper, good community templates |
| Vast.ai | ~$5/hr | ~$8/hr | Cheapest, less reliable |
| AWS p4d.24xlarge | ~$33/hr | — | Enterprise, most reliable |

### Recommended: Lambda Labs or RunPod

### Estimated Compute Budget:

| Stage | Hours | GPU | Cost |
|-------|-------|-----|------|
| Stage 1: CPT | 48-72 hrs | 4x A100 | $350-720 |
| Stage 2: SFT | 24-48 hrs | 4x A100 | $170-480 |
| Stage 3: DPO | 6-12 hrs | 4x A100 | $40-120 |
| Stage 4: RLHF (optional) | 30-48 hrs | 4x A100 | $210-480 |
| Evaluation & iteration | 12-24 hrs | 4x A100 | $85-240 |
| **Total** | **120-204 hrs** | | **$855-2,040** |

Add ~$500 for synthetic data generation (API costs).
Add ~$2,000-5,000 for human annotation (RLHF/eval).

### **Total project budget: $3,500-7,500**

---

## Timeline

| Week | Stage | Deliverable |
|------|-------|-------------|
| 1-2 | Data Factory | 50K synthetic conversations generated, validated |
| 2 | Data Factory | DPO preference pairs collected (automated) |
| 3 | CPT | Continued pretraining on Amharic corpus complete |
| 3-4 | SFT | Supervised fine-tuning with curriculum complete |
| 4 | DPO | DPO alignment complete |
| 4-5 | Eval | Benchmarks + human evaluation |
| 5 | RLHF | (Optional) RLHF training + eval |
| 5-6 | Deploy | Quantized model, API, voice integration |

**Total: 5-6 weeks to production-ready Tibeb.**

---

## Immediate Next Steps

1. **Add API credits** (~$500) → generate 50K synthetic conversations
2. **Select GPU provider** → Lambda Labs or RunPod account
3. **Download base model** → `huggingface-cli download Qwen/Qwen2.5-72B-Instruct`
4. **Prepare training scripts** → Adapt finetune_tibeb.py for multi-GPU DeepSpeed
5. **Recruit annotators** → 5-10 native Amharic speakers for DPO/RLHF

---

## File Structure (Production)

```
tibeb-training-1/
├── data/
│   ├── synthetic_qa/           # 50K generated conversations
│   ├── curated/                # Hand-curated financial data
│   ├── cpt_corpus/             # Amharic text for continued pretraining
│   ├── dpo/                    # Preference pairs
│   ├── rlhf/                   # Human annotation data
│   └── eval/                   # Evaluation benchmarks
├── scripts/
│   ├── generate_synthetic_data.py  # Conversation generator (updated)
│   ├── generate_fast.py            # Parallel generation
│   ├── generate_dpo_pairs.py       # DPO pair generator (NEW)
│   ├── merge_datasets.py           # Data pipeline
│   ├── train_cpt.py                # Continued pretraining (NEW)
│   ├── train_sft.py                # SFT with DeepSpeed (NEW)
│   ├── train_dpo.py                # DPO training (NEW)
│   ├── train_rlhf.py              # RLHF training (NEW)
│   ├── eval_benchmarks.py          # Automated benchmarks (NEW)
│   └── serve.py                    # vLLM inference server (NEW)
├── configs/
│   ├── cpt_config.yaml
│   ├── sft_config.yaml
│   ├── dpo_config.yaml
│   ├── deepspeed_z3.json
│   └── rlhf_config.yaml
├── docs/
│   ├── PRODUCTION_TRAINING_PLAN.md  # This document
│   └── RESEARCH_STRATEGY.md
└── eval/
    ├── financial_questions.jsonl    # 200 gold Q&A pairs
    ├── honorific_tests.jsonl        # 100 address form tests
    └── safety_prompts.jsonl         # 100 adversarial prompts
```
