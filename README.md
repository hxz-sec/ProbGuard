# ProbGuard


## Abstract


Recent research on Large Language Model (LLM) safety has widely adopted guardrails to identify unsafe model outputs. Existing guardrails usually formulate safety evaluation as a text-to-label classification task, which maps a discrete token sequence to a discrete safety label. However, this discrete paradigm discards the probabilistic signals from the output distribution and ignores that safety judgment, especially on early generation dynamics, is usually an uncertain problem. To address these limitations, we propose \textsc{ProbGuard}, a probability-based guardrail that leverages the target LLM's early output distributions to estimate the safety risk of its ongoing and future generation. This probabilistic risk signal enables early and adaptive safety intervention during generation, without waiting for the complete response.
Different from existing stream-based guardrail using hidden states for classification, \textsc{ProbGuard} can transfer to different model families since it only rely on the token probability vectors for prediction.  Experimental results show that \textsc{ProbGuard} achieves the lowest Brier Score and ECE across all nine model--dataset combinations, reducing the average Brier Score and ECE by 79.6\% and 71.7\%, respectively, compared with the strongest baseline.


## Code

### Preparation


**Datasets** </br>
| Dataset      | Link                                                         |
|-----------   |--------------------------------------------------------------|
| AdvBench     | https://github.com/llm-attacks/llm-attacks                   |
| DNA     | https://github.com/llm-attacks/llm-attacks                   |
| HarmBench    | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors |


