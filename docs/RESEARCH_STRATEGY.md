# Tibeb AI × EthioShare: Research & Competitive Strategy

**Purpose:** Beat Neway, reach all Ethiopians (including rural, feature-phone users), and make ESX investing compelling via Amharic voice AI + USSD.

---

## 1. What You Have (Current State)

### Tibeb Training Pipeline
- **Base model:** CohereForAI/aya-expanse-8b (strong Amharic)
- **Data:** EthioNLP, ALFFA voice transcriptions, Sujet Finance, synthetic Amharic financial QA
- **Topics:** T-Bills, ESX, FX, broker accounts, yield, diversification, ECMA rules, NBE rates
- **Spelling:** Authoritative Amharic (ሰላም, እርስዎ, ቲ-ቢል, etc.)
- **Honorifics:** Formal/informal, gender-aware (አቶ, ወይዘሮ, አንተ/አንቺ)

### Gaps vs. Production
- No voice (ASR/TTS) integration yet
- No USSD layer
- No ESX/Telebirr integration
- Not embedded in EthioShare app

---

## 2. Competitor: Neway (ESX Official App)

**Launched:** March 2026  
**Partner:** Infotech Private Limited (Broker Back Office, Order Management System)

### Neway Features
- Open trading accounts remotely (self-onboarding)
- Execute equity and fixed-income trades
- Monitor market trends and real-time portfolio
- Web + mobile app
- Direct integration with ESX broker ecosystem

### Neway Weaknesses (Your Opportunities)
| Gap | Your Edge |
|-----|-----------|
| **English-first, generic UX** | Tibeb: Amharic-native, honorifics, culturally tuned |
| **Smartphone + internet only** | USSD for feature phones, no data needed |
| **Text/UI only** | Voice: “ሰላም፣ አክሲዮን እንዴት እየሸጠሁ ነው?” — speak & listen |
| **Trading-focused only** | Tibeb: financial literacy, education, “how do I start?” |
| **No AI assistant** | Tibeb: conversational AI for investing questions |
| **Urban, literate users** | Voice + USSD = rural, low-literacy, elders |

---

## 3. Ethiopia Market Reality

### Mobile Penetration (2024)
- **Mobile ownership:** ~65–86% (men higher than women)
- **Smartphone:** ~15% overall (6% women, 18% men)
- **Feature phones:** Majority of users — no app store, no reliable internet
- **Data cost:** 5–8% of monthly income in some segments

### Implications
- **USSD is essential** for rural and low-income users
- **Voice is essential** for low literacy and elders
- **Amharic is non-negotiable** — English excludes most

---

## 4. USSD for ESX Investing

### African Precedents

**Nigeria**
- NGX: `*5474#` — real-time stock info, connect to brokers
- CSCS + MTN: `*7270#` — holdings, balances, portfolio (live May 2025)

**Kenya**
- Safaricom ZiiDi Trader — M-PESA + NSE: buy/sell shares via mobile money

### USSD Advantages
- Works on any GSM phone (2G+)
- No mobile data required
- Free or very low cost
- 45%+ of Sub-Saharan users still on feature phones

### Ethiopia Implementation Path
1. **Ethio Telecom Developer Portal:** `developer.ethiotelecom.et` — register for USSD/short-code APIs
2. **Telebirr integration:** Payments for ESX via Telebirr (already used for Ethio Telecom IPO)
3. **ESX broker partnership:** Neway uses broker back-office; you need a licensed broker to route orders
4. **USSD flow example:**
   ```
   *123# → Main menu (Amharic)
   1. አክሲዮን ዋጋ ተመልከት (Check share prices)
   2. ባለቤትነቴን ተመልከት (My portfolio)
   3. አክሲዮን ግዛ (Buy shares)
   4. ቲብብ ጥያቄ ጠይቅ (Ask Tibeb — voice callback or SMS)
   ```

---

## 5. Voice AI for Amharic

### Existing Options
| Provider | Use Case | Integration |
|----------|----------|-------------|
| **Addis AI** | Voice assistant, real-time WebSocket | `wss://relay.addisassistant.com/ws` — 16kHz PCM |
| **Abyssinica Speech** | Amharic ASR (speech → text) | API for transcription |
| **OpenMic.ai** | Voice AI agents, Amharic + cultural tuning | Customer service / sales |

### Tibeb Voice Strategy
1. **Option A:** Use Addis AI Realtime API for voice I/O, route text to Tibeb for financial logic
2. **Option B:** Build your own: Abyssinica ASR → Tibeb LLM → Amharic TTS (e.g. Coqui, or Ethiopian TTS models)
3. **Option C:** Fine-tune Tibeb for tool use (e.g. “check ESX price”, “place order”) and plug into Addis AI or custom voice pipeline

**Recommendation:** Start with Addis AI for voice; keep Tibeb as the brain for financial Q&A and intent extraction. Later, consider self-hosted ASR/TTS for cost and control.

---

## 6. Technical Integration Map

```
┌─────────────────────────────────────────────────────────────────┐
│                     EthioShare App                                │
├─────────────────────────────────────────────────────────────────┤
│  Smartphone (App)          │  Feature Phone (USSD)               │
│  ┌─────────────────────┐   │  ┌─────────────────────────────┐   │
│  │ Voice: Addis AI WS  │   │  │ *123# → Menu (Amharic)       │   │
│  │   → Tibeb LLM       │   │  │ → Tibeb SMS/IVR for Q&A      │   │
│  │   → ESX/Telebirr    │   │  │ → Telebirr for payments      │   │
│  └─────────────────────┘   │  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐            ┌─────────────────────┐
│ Tibeb (Aya 8B SFT)   │            │ USSD Gateway         │
│ - Financial Q&A      │            │ (Ethio Telecom API)  │
│ - Intent → actions   │            │ Telebirr API         │
└─────────────────────┘            └─────────────────────┘
         │                                    │
         └────────────────┬───────────────────┘
                          ▼
              ┌───────────────────────┐
              │ ESX Broker / CSD      │
              │ (via licensed broker) │
              └───────────────────────┘
```

---

## 7. How to Beat Neway

### 1. **Amharic-First, Voice-First**
- Neway: generic, likely English-heavy
- Tibeb: “ሰላም እርስዎ፣ ዛሬ ኢትዮ ቴሌኮም ዋጋ ስንት ነው?” — natural, respectful, educational

### 2. **Financial Literacy, Not Just Trading**
- Neway: trading UI
- Tibeb: “አክሲዮን ምንድነው? እንዴት እየገዛሁ ነው?” — explain, then help invest

### 3. **USSD = Rural + Feature Phones**
- Neway: smartphone + internet only
- Tibeb + USSD: 85%+ of mobile users can at least check prices, ask questions, get SMS replies

### 4. **Telebirr Integration**
- Ethio Telecom IPO already on Telebirr
- ESX + Telebirr for payments is a natural fit — partner with a broker who supports it

### 5. **Embedded in EthioShare**
- Neway: standalone ESX app
- Tibeb: inside EthioShare — one app for investing, community, education

### 6. **Elder-Friendly**
- Honorifics (እርስዎ, አቶ, ወይዘሮ)
- Voice: no typing
- USSD: no smartphone needed

---

## 8. Action Roadmap

### Phase 1: Strengthen Tibeb (Weeks 1–4)
- [ ] Add ESX-specific synthetic data (tickers, prices, “how to buy Ethio Telecom shares”)
- [ ] Add USSD-style short Q&A (“1. ዋጋ 2. ባለቤትነት 3. ግዛ”)
- [ ] Add voice-style prompts (“ሰላም፣ አክሲዮን እንዴት ይገዛል?”)
- [ ] Integrate Addis AI Realtime API for voice prototype

### Phase 2: EthioShare Integration (Weeks 4–8)
- [ ] Embed Tibeb in EthioShare app (API or in-app WebView)
- [ ] Connect to Telebirr for payments (developer.ethiotelecom.et)
- [ ] Partner with one licensed ESX broker for order routing

### Phase 3: USSD (Weeks 8–16)
- [ ] Register USSD short code via Ethio Telecom Developer Portal
- [ ] Design Amharic menu flow (prices, portfolio, buy, Tibeb Q&A)
- [ ] Implement Tibeb-as-SMS or IVR for “Ask Tibeb” option
- [ ] Integrate Telebirr for USSD payments

### Phase 4: Scale & Differentiate (Ongoing)
- [ ] Add more Ethiopian languages (Oromiffa, Tigrinya) if data allows
- [ ] Offline-capable Tibeb for low-connectivity
- [ ] Agent network: “Ask Tibeb” at Telebirr/HelloCash agents

---

## 9. Key Resources

| Resource | URL | Use |
|----------|-----|-----|
| Ethio Telecom Developer Portal | developer.ethiotelecom.et | Telebirr, USSD, APIs |
| Addis AI Realtime API | platform.addisassistant.com | Voice I/O |
| Abyssinica Speech | abyssinica.ai/asr | Amharic ASR |
| ESX Trading | esx.et | Market info, broker list |
| EthioNLP | huggingface.co/EthioNLP | Amharic NLP data |
| Aya Model | arxiv.org/abs/2402.08015 | Multilingual base |

---

## 10. Summary: Your Moat

| Dimension | Neway | Tibeb × EthioShare |
|-----------|-------|---------------------|
| Language | Likely English-first | Amharic-native, honorifics |
| Access | Smartphone + internet | App + USSD (feature phones) |
| Interaction | UI only | Voice + text + USSD |
| Focus | Trading | Literacy + education + trading |
| Reach | Urban, literate | Urban + rural, all literacy levels |
| Distribution | Standalone | Embedded in EthioShare |

**Bottom line:** Neway owns the official ESX trading channel. You own **Amharic financial AI + voice + USSD + EthioShare**. Partner with a broker for order execution; differentiate on language, accessibility, and education.
