#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "probguard/train_single_guard_v8_0.py"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
DEFAULT_DATA_FILE = PROJECT_ROOT / "examples/calibration_sample.jsonl"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"
DEFAULT_QWEN_EMBED = ""
DEFAULT_LOG = PROJECT_ROOT / "logs/probguard_stream.log"
DEFAULT_JSONL = PROJECT_ROOT / "outputs/probguard_stream_calibration_compare.jsonl"


def load_train_module():
    spec = importlib.util.spec_from_file_location("probguard_train_v8_stream", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import train script: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_logger(log_path: Path, verbose: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("probguard_stream")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def pick_gpu(requested_gpu: str, logger: logging.Logger) -> torch.device:
    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    if requested_gpu != "auto":
        gpu_id = int(requested_gpu)
        torch.cuda.set_device(gpu_id)
        logger.info("Selected cuda:%d by argument.", gpu_id)
        return torch.device(f"cuda:{gpu_id}")

    best_gpu = 0
    best_free = -1
    for gpu_id in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu_id):
            free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info("GPU %d free=%.2fGiB total=%.2fGiB", gpu_id, free_mem / 1024**3, total_mem / 1024**3)
        if free_mem > best_free:
            best_gpu = gpu_id
            best_free = free_mem
    torch.cuda.set_device(best_gpu)
    logger.info("Auto selected cuda:%d.", best_gpu)
    return torch.device(f"cuda:{best_gpu}")


def model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def find_latest_checkpoint(root: Path) -> Path:
    candidates = sorted(
        root.glob("**/best_checkpoint/probguard_heads.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best_checkpoint/probguard_heads.pt found under {root}")
    return candidates[0].parent


def load_embedding_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        tensors = [value for value in obj.values() if isinstance(value, torch.Tensor) and value.ndim == 2]
        if not tensors:
            raise RuntimeError(f"No 2D tensor found in {path}")
        obj = tensors[0]
    if not isinstance(obj, torch.Tensor) or obj.ndim != 2:
        raise RuntimeError(f"Embedding file must contain a 2D tensor: {path}")
    return obj.detach().to(device=device, dtype=dtype)


def load_probguard(checkpoint_dir: Path, train_mod, device: torch.device, dtype: torch.dtype, qwen_embed_path: Path, logger):
    metadata = torch.load(checkpoint_dir / "probguard_heads.pt", map_location="cpu")
    if "projector" in metadata:
        raise RuntimeError(f"{checkpoint_dir} is an old projector checkpoint. Use a no-projector checkpoint.")

    args = metadata.get("args", {})
    guard_model_path = args.get("guard_model", str(DEFAULT_QWEN_MODEL))
    requested_embed_weight = str(qwen_embed_path)
    if requested_embed_weight == ".":
        requested_embed_weight = ""
    generation_embed_weight = requested_embed_weight or str(args.get("generation_embed_weight", ""))
    logger.info("Loading no-projector ProbGuard checkpoint: %s", checkpoint_dir)
    model_dir = checkpoint_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Merged ProbGuard model not found: {model_dir}. Train/export the checkpoint again.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    guard_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map={"": device} if device.type == "cuda" else None,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    guard_model.config.use_cache = False
    guard_model = guard_model.to(device)
    guard_model.eval()

    hidden_size = guard_model.get_input_embeddings().weight.shape[1]
    heads = train_mod.ProbGuardHeads(hidden_size, len(train_mod.CATEGORIES)).to(device=device)
    heads.load_state_dict(metadata["heads"])
    heads.eval()
    template_embeds = train_mod.encode_static_template(tokenizer, guard_model.get_input_embeddings(), device, dtype)
    if generation_embed_weight:
        qwen_weight = load_embedding_tensor(Path(generation_embed_weight), device, dtype)
        logger.info("Loaded Qwen embed weight: shape=%s", tuple(qwen_weight.shape))
    else:
        qwen_weight = guard_model.get_input_embeddings().weight.detach().to(device=device, dtype=dtype)
        logger.info("Using guard input embedding as Qwen/token soft embedding: shape=%s", tuple(qwen_weight.shape))
    return guard_model, tokenizer, heads, template_embeds, qwen_weight


def load_qwen_tokenizer(path: str, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def iter_rows(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if limit and idx >= limit:
                break
            line = line.strip()
            if line:
                yield idx, json.loads(line)


def choose_prefix_steps(row: dict[str, Any], k: int, sample_index: int) -> list[dict[str, Any]] | None:
    details = row.get("prefix_generation_details") or {}
    candidates = details.get(str(k))
    if not isinstance(candidates, list) or not candidates:
        return None
    prefix = candidates[sample_index % len(candidates)]
    if not isinstance(prefix, list) or len(prefix) < k:
        return None
    steps = prefix[:k]
    for step in steps:
        if not isinstance(step, dict):
            return None
        tokens = step.get("topk_tokens")
        probs = step.get("topk_probs")
        if not isinstance(tokens, list) or not isinstance(probs, list) or len(tokens) != len(probs) or not tokens:
            return None
    return steps


@torch.inference_mode()
def predict_c(
    train_mod,
    probguard,
    qwen_tokenizer,
    prompt: str,
    steps: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    max_prompt_len: int,
    token_id_cache: dict[str, list[int]],
):
    guard_model, guard_tokenizer, heads, template_embeds, qwen_weight = probguard
    sample = train_mod.ProbGuardSample(
        prompt=prompt,
        k=len(steps),
        target_c=0.0,
        category_label=train_mod.NONE_CATEGORY_ID,
        steps=steps,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    inputs_embeds, attention_mask, last_indices, _, _ = train_mod.build_probguard_batch(
        batch=[sample],
        tokenizer=guard_tokenizer,
        guard_embedding=guard_model.get_input_embeddings(),
        template_embeds=template_embeds,
        generation_tokenizer=qwen_tokenizer,
        embedding_tokenizer=guard_tokenizer,
        generation_weight=qwen_weight,
        device=device,
        model_dtype=dtype,
        max_prompt_len=max_prompt_len,
        token_id_cache=token_id_cache,
    )
    pred_prob, category_logits = train_mod.forward_probguard(guard_model, heads, inputs_embeds, attention_mask, last_indices)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    category_id = int(category_logits.argmax(dim=-1)[0].detach().cpu().item())
    return float(pred_prob[0].detach().float().cpu().item()), train_mod.CATEGORIES[category_id], elapsed_ms


def compare_file(args, train_mod, probguard, qwen_tokenizer, device: torch.device, dtype: torch.dtype, logger):
    jsonl_path = Path(args.jsonl_log)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    rows_seen = 0
    all_abs_diffs: list[float] = []

    with jsonl_path.open("w", encoding="utf-8") as out:
        for row_idx, row in iter_rows(Path(args.data_file), args.num_samples):
            rows_seen += 1
            sample_id = row.get("id", row_idx)
            prompt = str(row.get("goal") or row.get("harmful") or row.get("prompt") or "")
            targets = row.get("calibration_probabilities") or {}
            token_id_cache: dict[str, list[int]] = {}
            sample_diffs: list[float] = []

            print("\n" + "=" * 100)
            print(f"Sample {rows_seen} | id={sample_id} | category={row.get('primary_category')}")
            print(f"goal_preview={prompt[:180].replace(chr(10), ' ')}")
            print("k | pred_C | target_C | abs_diff | category | guard_ms")

            for k in range(args.k_min, args.k_max + 1):
                if str(k) not in targets:
                    continue
                steps = choose_prefix_steps(row, k, args.prefix_sample_index)
                if steps is None:
                    logger.warning("Skipping id=%s k=%d because prefix_generation_details is missing or invalid.", sample_id, k)
                    continue
                pred_c, category, elapsed_ms = predict_c(
                    train_mod=train_mod,
                    probguard=probguard,
                    qwen_tokenizer=qwen_tokenizer,
                    prompt=prompt,
                    steps=steps,
                    device=device,
                    dtype=dtype,
                    max_prompt_len=args.max_prompt_len,
                    token_id_cache=token_id_cache,
                )
                target_c = float(targets[str(k)])
                abs_diff = abs(pred_c - target_c)
                sample_diffs.append(abs_diff)
                all_abs_diffs.append(abs_diff)
                print(f"{k:02d} | {pred_c:.4f} | {target_c:.4f} | {abs_diff:.4f} | {category} | {elapsed_ms:.1f}")
                out.write(
                    json.dumps(
                        {
                            "row_index": row_idx,
                            "id": sample_id,
                            "k": k,
                            "goal": prompt,
                            "target_C": target_c,
                            "pred_C": pred_c,
                            "abs_diff": abs_diff,
                            "category": category,
                            "guard_ms": elapsed_ms,
                            "probguard_input_type": "qwen_prefix_generation_details_topk_tokens_probs",
                            "prefix_sample_index": args.prefix_sample_index,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if sample_diffs:
                print(f"Sample summary | mean_abs_diff={sum(sample_diffs) / len(sample_diffs):.4f} | max_abs_diff={max(sample_diffs):.4f}")

    if all_abs_diffs:
        print("\n" + "=" * 100)
        print(
            "Overall summary | "
            f"samples={rows_seen} compared_steps={len(all_abs_diffs)} "
            f"mean_abs_diff={sum(all_abs_diffs) / len(all_abs_diffs):.4f} "
            f"max_abs_diff={max(all_abs_diffs):.4f}"
        )
    print(f"Structured comparison log: {jsonl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare streaming ProbGuard predictions with stored Qwen calibration targets.")
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--checkpoint-root", type=str, default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--data-file", type=str, default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--qwen-model", type=str, default=str(DEFAULT_QWEN_MODEL))
    parser.add_argument("--qwen-embed-weight", type=str, default=str(DEFAULT_QWEN_EMBED))
    parser.add_argument("--local-files-only", action="store_true", help="Load target tokenizer from the local cache only.")
    parser.add_argument("--gpu", type=str, default="auto")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--prefix-sample-index", type=int, default=0)
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--log-file", type=str, default=str(DEFAULT_LOG))
    parser.add_argument("--jsonl-log", type=str, default=str(DEFAULT_JSONL))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logger = setup_logger(Path(args.log_file), args.verbose)
    train_mod = load_train_module()
    device = pick_gpu(args.gpu, logger)
    dtype = model_dtype(device)
    checkpoint_dir = find_latest_checkpoint(Path(args.checkpoint_root)) if args.checkpoint == "auto" else Path(args.checkpoint)
    probguard = load_probguard(checkpoint_dir, train_mod, device, dtype, Path(args.qwen_embed_weight), logger)
    qwen_tokenizer = load_qwen_tokenizer(args.qwen_model, args.local_files_only)
    compare_file(args, train_mod, probguard, qwen_tokenizer, device, dtype, logger)
    print(f"Runtime log: {Path(args.log_file)}")


if __name__ == "__main__":
    main()
