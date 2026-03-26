# Human-Authored Training Data

Write conversations in the JSONL files in this directory.
Each line is one conversation turn.

## Format
```json
{"instruction": "user question in Amharic", "input": "", "output": "ideal Tibeb response", "source": "human_nahom", "address_form": "informal_male"}
```

## Files
- `financial_qa.jsonl` — Financial questions and answers
- `general_qa.jsonl` — General Amharic conversations
- `corrections.jsonl` — Fixed versions of bad synthetic data
- `voice_transcripts.jsonl` — Transcribed voice recordings

## Tips
- Write how you actually speak, not textbook Amharic
- Include questions your family/friends would ask
- Mix formal (እርስዎ) and informal (አንተ/አንቺ)
- Include "I don't know" answers where appropriate
- Include scam/safety scenarios
