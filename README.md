# ProbGuard

[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#license)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/hxz-sec/ProbGuard)

Official repository for **ProbGuard**, a probability-based guardrail for forecasting unsafe continuations during LLM generation.

ProbGuard moves safety monitoring from post-hoc response classification to generation-time risk estimation. It reads the target LLM's early next-token probability distributions and predicts whether the ongoing response is likely to become unsafe before harmful content is fully emitted.

- **Model**: https://huggingface.co/hxz-sec/ProbGuard-8b

<p align="center">
  <img src="assets/problem_.jpg" alt="Comparison of post-hoc guardrails, streaming monitors, probes, and ProbGuard" width="95%">
</p>


## Abstract

Existing LLM guardrails usually formulate safety evaluation as a text-to-label classification problem. They map a completed response, or a generated prefix, to a discrete safety label. This discrete paradigm discards the probabilistic signal exposed by the target LLM's output distribution and treats an inherently uncertain early-generation problem as a hard classification task.

ProbGuard addresses this limitation with a **prob-to-prob** formulation. Instead of classifying only observed text, ProbGuard leverages the target LLM's early output probability distributions to estimate the calibrated risk that the ongoing generation will eventually become unsafe. Because this signal is available during decoding, ProbGuard enables early and adaptive intervention without waiting for the complete response. Because it relies on output-space probability vectors rather than hidden states, it can transfer across model families without model-specific probes.

**Figure 1.** Jailbreak example.  
![Jailbreak example](assets/problem_.jpg)

**Figure 2.** Workflow of ProbGuard framework.  
![Workflow](assets/workflow.png)






## Installation

Create a Python environment and install the core dependencies:

```bash
conda create -n probguard python=3.11 -y
conda activate probguard

```

## Model and Data Preparation

ProbGuard training and inference require access to:

- a base ProbGuard backbone such as Qwen3;
- a target LLM that exposes top-k next-token probabilities;
- prefix-calibration JSONL files;
- the target-model embedding tensor when using cached embedding weights.

The main data-preparation entry point is:

```text
data_prepare/prefix_calibration.py
```

It collects prefix states and calibration signals used by the downstream ProbGuard trainer. Pass local paths for the target LLM and RoBERTa safety judge as needed.

## Training ProbGuard

The main trainer is:

```text
probguard/train_single_guard_v8_0.py
```

Example command:

```bash
python probguard/train_single_guard_v8_0.py \
  --model-name ProbGuard-8B-mixed \
  --guard-model /path/to/Qwen3-8B \
  --generation-model /path/to/Qwen3-8B \
  --generation-tokenizer /path/to/Qwen3-8B \
  --generation-embed-weight /path/to/Qwen3_embed_weight.pt \
  --epochs 4 \
  --batch-size 16 \
  --k-min 5 \
  --k-max 15 \
  --gradient-checkpointing \
  --best-metric loss \
  --keep-only-best
```

## Evaluating ProbGuard

The main evaluation script is:

```text
probguard/eval_probguard_infer.py
```

Example command:

```bash
python probguard/eval_probguard_infer.py \
  --train-script probguard/train_single_guard_v8_0.py \
  --checkpoint /path/to/best_checkpoint \
  --output-dir outputs/eval \
  --save-output-files \
  --gpu 0 \
  --batch-size 64 \
  --k-values 10
```

The evaluation pipeline reports calibration quality with Brier Score and Expected Calibration Error, and can save per-dataset prediction files for further analysis.



## Inference

ProbGuard is not a standalone chat model. It should not be used through plain `generate()` as an assistant.

Correct inference requires:

- the target prompt;
- the target LLM's early top-k probability vectors;
- the merged ProbGuard model;
- `probguard_heads.pt`;
- the ProbGuard inference utilities.

High-level pseudocode:

```python
prompt = "..."
topk_steps = target_llm.collect_topk_prefix_distribution(prompt)

risk, category = probguard.predict(
    prompt=prompt,
    topk_steps=topk_steps,
)

if risk >= threshold:
    stop_or_redirect_generation()
```

## Harmful Categories

ProbGuard predicts one primary category from:

```text
Toxicity, Hate, Violence, Sexual, Harm, Drugs, Conflict,
Illegal, Medical, Extremism, None
```

`None` is used when the continuation is likely to remain safe.

<!-- ## Results

Across PKU, WildGuard, and S-Eval, ProbGuard consistently improves probability calibration for early unsafe-continuation forecasting compared with full-response guardrails, streaming monitors, confidence elicitation, and hidden-state probes.

The paper reports that ProbGuard achieves the lowest Brier Score and ECE across all evaluated model-dataset combinations, and maintains strong jailbreak interception under AdvBench and HarmBench attack settings. Full quantitative results are provided in the paper.

<p align="center">
  <img src="assets/table3.png" alt="Efficiency comparison between ProbGuard and baselines" width="85%">
</p>

ProbGuard also benefits from accumulating more prefix probability evidence during generation. The prefix-length analysis shows how calibration changes as the guard observes longer early-generation states. -->






<!-- <p align="center">
  <img src="assets/prefix_lenth.jpg" alt="ProbGuard calibration across prefix lengths" width="85%">
</p> -->

<!-- ## Citation

If you find this repository useful, please cite our work:

```bibtex
@misc{probguard2026,
  title        = {ProbGuard},
  author       = {Anonymous},
  year         = {2026},
  note         = {Probability-based guardrail for unsafe continuation forecasting}
}
``` -->

## License

This project is released for research use. Please check the final repository license before redistribution or commercial use.
