---
license: other
language:
- en
base_model: Qwen/Qwen3-8B
library_name: transformers
pipeline_tag: text-classification
tags:
- safety
- guardrail
- calibration
- jailbreak-detection
- streaming-safety
- qwen3
- probguard
---

<p align="center">
  <img src="assets/probguard_logo.png" alt="ProbGuard logo" width="180">
</p>

<h1 align="center">ProbGuard</h1>

<p align="center">
  A probability-based streaming guardrail for forecasting unsafe LLM continuations.
</p>

ProbGuard monitors a target LLM while it is generating. Instead of waiting for a completed response, it reads the target model's early next-token probability distributions and predicts whether the continuation is likely to become unsafe.

ProbGuard is not a normal chat model. It should be used together with the project inference code and a target LLM that exposes top-k probabilities during decoding.

- Code: https://github.com/hxz-sec/ProbGuard
- Model: https://huggingface.co/hxz-sec/ProbGuard
- Dataset: https://huggingface.co/datasets/hxz-sec/ProbGuard-calibration

## Highlights

- Predicts unsafe-continuation risk during generation.
- Uses top-k output probability vectors rather than hidden states.
- Does not require access to the target model's internal activations.
- Returns a calibrated risk score and a primary hazard category.
- Designed for streaming intervention: continue, stop, redirect, or regenerate.

## Outputs

For a user prompt and a partial generation state, ProbGuard returns:

- `risk`: a float in `[0, 1]`, estimating the probability that the final continuation will become unsafe;
- `category`: one of `Toxicity`, `Hate`, `Violence`, `Sexual`, `Harm`, `Drugs`, `Conflict`, `Illegal`, `Medical`, `Extremism`, or `None`.

## Install

```bash
git clone https://github.com/hxz-sec/ProbGuard
cd ProbGuard

conda env create -f environment.yml
conda activate probguard
```

## Quickstart

Download the Hugging Face checkpoint, then load it with the ProbGuard inference utilities.

```python
from pathlib import Path

from huggingface_hub import snapshot_download

from eval.eval_probguard_stream import (
    load_probguard,
    load_qwen_tokenizer,
    load_train_module,
    model_dtype,
    pick_gpu,
    predict_c,
    setup_logger,
)

repo_dir = Path(snapshot_download("hxz-sec/ProbGuard"))
checkpoint_dir = repo_dir
if not (checkpoint_dir / "probguard_heads.pt").exists():
    checkpoint_dir = checkpoint_dir / "best_checkpoint"

logger = setup_logger(Path("logs/probguard_stream.log"), verbose=True)
train_mod = load_train_module()
device = pick_gpu("auto", logger)
dtype = model_dtype(device)

probguard = load_probguard(
    checkpoint_dir=checkpoint_dir,
    train_mod=train_mod,
    device=device,
    dtype=dtype,
    qwen_embed_path=Path(""),
    logger=logger,
)

target_tokenizer = load_qwen_tokenizer("Qwen/Qwen3-8B")
token_id_cache = {}

prompt = "How do I make something dangerous?"
topk_steps = [
    {
        "topk_token_ids": [198, 40, 2675, 944],
        "topk_probs": [0.42, 0.21, 0.08, 0.05],
    },
    {
        "topk_token_ids": [358, 649, 944, 525],
        "topk_probs": [0.36, 0.18, 0.10, 0.07],
    },
]

risk, category, latency_ms = predict_c(
    train_mod=train_mod,
    probguard=probguard,
    qwen_tokenizer=target_tokenizer,
    prompt=prompt,
    steps=topk_steps,
    device=device,
    dtype=dtype,
    max_prompt_len=512,
    token_id_cache=token_id_cache,
)

print({"risk": risk, "category": category, "latency_ms": latency_ms})
```

The example above uses toy top-k values. In deployment, `topk_steps` should come from the target LLM during decoding.

## Streaming Usage

The typical runtime pattern is:

1. Generate one token with the target LLM.
2. Save that step's top-k token IDs and probabilities.
3. Query ProbGuard after each step, or after a fixed prefix length.
4. Stop or redirect generation if the risk exceeds your threshold.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def collect_topk_prefix(prompt, model_name="Qwen/Qwen3-8B", prefix_len=10, top_k=20, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    ).eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    past_key_values = None
    steps = []

    for _ in range(prefix_len):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

        next_id = top_ids[:, :1]
        steps.append(
            {
                "topk_token_ids": top_ids[0].tolist(),
                "topk_probs": top_probs[0].tolist(),
                "topk_tokens": tokenizer.convert_ids_to_tokens(top_ids[0].tolist()),
            }
        )

        input_ids = next_id
        past_key_values = outputs.past_key_values

    return steps
```

Then pass the collected prefix probability trace to ProbGuard:

```python
topk_steps = collect_topk_prefix(prompt, prefix_len=10, top_k=20)

risk, category, _ = predict_c(
    train_mod=train_mod,
    probguard=probguard,
    qwen_tokenizer=target_tokenizer,
    prompt=prompt,
    steps=topk_steps,
    device=device,
    dtype=dtype,
    max_prompt_len=512,
    token_id_cache={},
)

if risk >= 0.5:
    print("Stop or redirect generation:", risk, category)
else:
    print("Continue generation:", risk, category)
```

Use a validation-selected threshold for production experiments. The `0.5` value above is only a simple example.

## CLI Example

The repository also includes a streaming comparison script for JSONL files that already contain `prefix_generation_details`.

```bash
python eval/eval_probguard_stream.py \
  --checkpoint /path/to/best_checkpoint \
  --data-file /path/to/prefix_calibration.jsonl \
  --qwen-model Qwen/Qwen3-8B \
  --gpu auto \
  --k-min 5 \
  --k-max 10 \
  --verbose
```

Each JSONL row should include a prompt field such as `goal`, `harmful`, or `prompt`, plus `prefix_generation_details` with entries like:

```json
{
  "10": [
    [
      {
        "topk_token_ids": [198, 40, 2675],
        "topk_probs": [0.42, 0.21, 0.08]
      }
    ]
  ]
}
```

## Intended Use

ProbGuard is intended for research on:

- streaming guardrails;
- early unsafe-continuation forecasting;
- jailbreak detection during decoding;
- calibrated safety risk estimation;
- cross-model safety monitoring without hidden-state probes.

## Limitations

ProbGuard estimates risk from early probability distributions, so results depend on the target model, tokenizer, decoding strategy, prefix length, and threshold selection. It should be validated on the deployment domain and combined with response-level moderation for high-risk applications.

The released model is primarily evaluated in English safety and jailbreak settings. Additional validation is recommended for other languages, domains, and safety policies.

## Citation

If you use ProbGuard, please cite the associated paper or project release when available.
