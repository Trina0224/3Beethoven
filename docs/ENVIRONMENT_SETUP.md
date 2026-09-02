# 3Beethoven environment setup

This document records the reproducible environment used for the first distillation experiment.

## Services

- Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct`
- Hugging Face read token stored as `HF_TOKEN`
- OpenRouter account with an API key stored as `OPENROUTER_API_KEY`
- Kaggle notebook with GPU enabled

## Kaggle Secrets

Create these two secrets in Kaggle. Never hard-code or print them.

- `HF_TOKEN`
- `OPENROUTER_API_KEY`

Recommended secret loading snippet:

```python
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
OPENROUTER_API_KEY = secrets.get_secret("OPENROUTER_API_KEY")
```

## Teacher smoke test

The teacher used for the first experiment is Meta Llama 3.3 70B Instruct via OpenRouter.

```python
import requests

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
payload = {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "messages": [
        {
            "role": "user",
            "content": "Who composed the Goldberg Variations? Answer in one sentence.",
        }
    ],
    "temperature": 0.2,
}
response = requests.post(url, headers=headers, json=payload, timeout=60)
print("HTTP status:", response.status_code)
print(response.json()["choices"][0]["message"]["content"])
```

Expected result: HTTP 200 and an answer naming Johann Sebastian Bach.

## Hugging Face access smoke test

```python
from huggingface_hub import HfApi

api = HfApi()
me = api.whoami(token=HF_TOKEN)
print("HF user:", me["name"])
```

If a newly approved gated model still returns 401/403 with an older token, create a new read token after access is granted and update the Kaggle secret.

## Load the student model

Enable the Kaggle GPU before running this section.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

## First student smoke test

```python
prompt = """
You are answering a classical music question.

Question:
Which musical period does Johann Sebastian Bach belong to, and name one structural feature commonly associated with his music?

Answer concisely.
"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

## Experimental discipline

- Freeze benchmark questions before teacher-data generation.
- Never put benchmark items into training data.
- Record baseline results before fine-tuning.
- Keep teacher-generation prompts, filtering rules, training configuration, and post-training evaluation reproducible.
- Do not commit model weights, API keys, generated secrets, NDA material, or company-internal data.
