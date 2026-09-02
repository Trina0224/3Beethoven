"""Generate the first 3Beethoven teacher-data pilot with Llama 3.3 70B via OpenRouter.

This script is intentionally small: 60 examples only. Inspect the generated data manually
before scaling up. NEVER include frozen benchmark questions or near-duplicates.

Requires Kaggle secret: OPENROUTER_API_KEY
Output: /kaggle/working/teacher_pilot_v0_1.jsonl
"""

import json
import time
import requests
from kaggle_secrets import UserSecretsClient

OPENROUTER_API_KEY = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-3.3-70b-instruct"
URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = "/kaggle/working/teacher_pilot_v0_1.jsonl"

PLAN = {
    "harmony_counterpoint": 30,
    "orchestration": 12,
    "form_analysis": 10,
    "history_context": 5,
    "style_comparison": 3,
}

SYSTEM = """You are a conservatory-level classical-music instructor creating synthetic training data for a smaller language model.

Create rigorous but teachable examples. Avoid trivia that a generic model is very likely to know already. Favor conceptual distinctions, analytical reasoning, terminology in context, and common misconceptions.

CRITICAL RULES:
- Do not copy or closely paraphrase benchmark questions.
- Do not assume access to scores or audio unless the prompt explicitly provides enough information.
- Prefer established music-theory/musicology facts over subjective aesthetic judgments.
- Be concise and factual.
- If a topic is genuinely disputed or terminology varies by tradition, say so rather than pretending there is one universal answer.
- Output valid JSON only, with no Markdown fences.
"""


def call_teacher(category, index):
    prompt = f"""Create ONE training example for category: {category}.

Return this exact JSON object schema:
{{
  "category": "{category}",
  "question": "a fresh classical-music question",
  "answer": "a correct concise expert answer",
  "explanation": "2-5 sentences explaining why, including the decisive concept",
  "common_mistake": "one plausible misconception and how to avoid it",
  "difficulty": "intermediate" or "advanced"
}}

This is example {index}. Make it materially different from common introductory trivia and from earlier examples.
"""

    r = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 700,
        },
        timeout=90,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    obj = json.loads(text)
    obj["teacher_model"] = MODEL
    return obj


rows = []
for category, count in PLAN.items():
    print(f"\nGenerating {count} examples for {category}...")
    for i in range(1, count + 1):
        try:
            item = call_teacher(category, i)
            rows.append(item)
            print(f"  {i:02d}/{count} OK - {item['question'][:70]}")
        except Exception as e:
            print(f"  {i:02d}/{count} FAILED: {e}")
        time.sleep(0.35)

with open(OUT, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\n==============================")
print("TEACHER PILOT COMPLETE")
print("==============================")
print("Generated:", len(rows), "/", sum(PLAN.values()))
print("Saved to:", OUT)

# Print a few samples for manual inspection without exposing the API key.
for i, row in enumerate(rows[:5], 1):
    print(f"\n--- SAMPLE {i} ---")
    print("Category:", row.get("category"))
    print("Q:", row.get("question"))
    print("A:", row.get("answer"))
    print("Why:", row.get("explanation"))
    print("Common mistake:", row.get("common_mistake"))
