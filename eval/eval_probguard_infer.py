#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_single_guard_v8_0.py"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/eval"
DEFAULT_GENERATION_EMBED_WEIGHT = ""
DEFAULT_GUARD_MODEL = "Qwen/Qwen3-8B"

CHECKPOINT_ALIASES = {
    "8b": "ProbGuard-8B-mixed",
    "4b": "ProbGuard-4B-mixed",
    "0.6b": "ProbGuard-0.6B-mixed",
}

DATASET_PAIRS = {
    "pku__qwen3": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample1/classified_pku_1000_Qwen3-8B_calibration_sample1_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_pku_1000_Qwen3-8B_calibration_sample16_1000.jsonl",
    ),
    "SEval__qwen3": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample1/classified_SEval_1000_Qwen3-8B_calibration_sample1_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_SEval_1000_Qwen3-8B_calibration_sample16_1000.jsonl",
    ),
    "wildGuard__qwen3": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample1/classified_wildGuard_1000_Qwen3-8B_calibration_sample1_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_wildGuard_1000_Qwen3-8B_calibration_sample16_1000.jsonl",
    ),
    "pku__llama3": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample1/classified_pku_1000_llama-3-8B-Instruct_calibration_sample1_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_pku_1000_llama-3-8B-Instruct_calibration_sample16_1000.jsonl",
    ),
    "SEval__llama3": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample1/classified_SEval_1000_llama-3-8B-Instruct_calibration_sample1_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_SEval_1000_llama-3-8B-Instruct_calibration_sample16_1000.jsonl",
    ),
    "wildGuard__llama3": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample1/classified_wildGuard_1000_llama-3-8B-Instruct_calibration_sample1_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_wildGuard_1000_llama-3-8B-Instruct_calibration_sample16_1000.jsonl",
    ),
    # Gemma currently has only sample16 files; those rows also carry prefix_generation_details.
    "pku__gemma2": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_pku_1000_Gemma-2-9b-it_calibration_sample16_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_pku_1000_Gemma-2-9b-it_calibration_sample16_1000.jsonl",
    ),
    "SEval__gemma2": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_SEval_1000_Gemma-2-9b-it_calibration_sample16_1000.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_SEval_1000_Gemma-2-9b-it_calibration_sample16_1000.jsonl",
    ),
    "wildGuard__gemma2": (
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_wildGuard_1000_Gemma-2-9b-it_calibration.jsonl",
        PROJECT_ROOT / "data/prefix_calibration/eval/jsonl_sample16/classified_wildGuard_1000_Gemma-2-9b-it_calibration.jsonl",
    ),
}


@dataclass
class EvalItem:
    dataset: str
    sample_id: int | str
    k: int
    prompt: str
    target_c: float
    category_label: int
    steps: list[dict[str, Any]]


def load_train_module(train_script: Path = TRAIN_SCRIPT):
    module_name = f"probguard_train_{train_script.stem.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import train script: {train_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("probguard_eval")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def pick_gpu(gpu: str, logger: logging.Logger) -> torch.device:
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available; using CPU.")
        return torch.device("cpu")

    if gpu != "auto":
        gpu_id = int(gpu)
        torch.cuda.set_device(gpu_id)
        logger.info("Selected GPU by argument: cuda:%d", gpu_id)
        return torch.device(f"cuda:{gpu_id}")

    best_gpu = 0
    best_free = -1
    for gpu_id in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu_id):
            free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info("GPU %d memory: free=%.2f GiB total=%.2f GiB", gpu_id, free_mem / 1024**3, total_mem / 1024**3)
        if free_mem > best_free:
            best_gpu = gpu_id
            best_free = free_mem
    torch.cuda.set_device(best_gpu)
    logger.info("Auto selected GPU: cuda:%d", best_gpu)
    return torch.device(f"cuda:{best_gpu}")


def find_latest_checkpoint(root: Path) -> Path:
    candidates = sorted(
        root.glob("**/best_checkpoint/probguard_heads.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best_checkpoint/probguard_heads.pt found under {root}")
    return candidates[0].parent


def resolve_checkpoint_dir(checkpoint_root: Path, checkpoint: str) -> Path:
    if checkpoint == "auto":
        return find_latest_checkpoint(checkpoint_root)
    alias = CHECKPOINT_ALIASES.get(checkpoint.lower())
    if alias is not None:
        return find_latest_checkpoint(checkpoint_root / alias)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.name == "best_checkpoint":
        return checkpoint_path
    if (checkpoint_path / "probguard_heads.pt").exists():
        return checkpoint_path
    return find_latest_checkpoint(checkpoint_path)


def resolve_checkpoint_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    checkpoint_root = Path(args.checkpoint_root)
    if args.checkpoint == "all":
        return [
            (alias, resolve_checkpoint_dir(checkpoint_root, alias))
            for alias in ("8b", "4b", "0.6b")
        ]
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_root, args.checkpoint)
    checkpoint_name = args.checkpoint if args.checkpoint in CHECKPOINT_ALIASES else checkpoint_dir.parent.parent.name
    return [(checkpoint_name, checkpoint_dir)]


def checkpoint_log_path(log_file: str, run_output_dir: Path, checkpoint_name: str, checkpoint_count: int) -> Path:
    if not log_file:
        return run_output_dir / f"{checkpoint_name}.log"
    path = Path(log_file)
    if checkpoint_count == 1:
        return path
    suffix = path.suffix or ".log"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_{checkpoint_name}{suffix}")


def same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return str(left).rstrip("/") == str(right).rstrip("/")


def load_generation_embedding(
    model_path: str,
    embed_weight_path: str,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger,
    local_files_only: bool = False,
) -> torch.Tensor:
    if embed_weight_path:
        path = Path(embed_weight_path)
        logger.info("Loading generation embedding from tensor file: %s", path)
        weight_obj = torch.load(path, map_location="cpu")
        if isinstance(weight_obj, dict):
            tensor_candidates = [value for value in weight_obj.values() if isinstance(value, torch.Tensor) and value.ndim == 2]
            if not tensor_candidates:
                raise RuntimeError(f"No 2D tensor found in embedding weight file: {path}")
            weight_obj = tensor_candidates[0]
        if not isinstance(weight_obj, torch.Tensor) or weight_obj.ndim != 2:
            raise RuntimeError(f"Embedding weight file must contain a 2D tensor: {path}")
        return weight_obj.detach().to(device=device, dtype=dtype)

    logger.info("Loading generation embedding from separate model: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device} if device.type == "cuda" else None,
        local_files_only=local_files_only,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    weight = model.get_input_embeddings().weight.detach().to(device=device, dtype=dtype).clone()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return weight


def load_probguard(
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    train_mod,
    logger: logging.Logger,
    generation_model_override: str | None = None,
    generation_tokenizer_override: str | None = None,
    generation_embed_weight_override: str | None = None,
    local_files_only: bool = False,
):
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    metadata = torch.load(checkpoint_dir / "probguard_heads.pt", map_location="cpu")
    args = metadata.get("args", {})
    guard_model_path = args.get("guard_model", DEFAULT_GUARD_MODEL)
    generation_model_path = generation_model_override or args.get("generation_model", guard_model_path)
    generation_tokenizer_path = generation_tokenizer_override or args.get("generation_tokenizer", generation_model_path)
    generation_embed_weight_path = generation_embed_weight_override
    if generation_embed_weight_path is None:
        generation_embed_weight_path = args.get("generation_embed_weight", str(DEFAULT_GENERATION_EMBED_WEIGHT))
    if generation_model_override:
        logger.info("Generation model override enabled: %s", generation_model_override)
    if generation_tokenizer_override:
        logger.info("Generation tokenizer override enabled: %s", generation_tokenizer_override)
    if generation_embed_weight_override:
        logger.info("Generation embedding override enabled: %s", generation_embed_weight_override)

    model_dir = checkpoint_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Merged ProbGuard model not found: {model_dir}. Train/export the checkpoint again.")

    logger.info("Loading merged ProbGuard tokenizer/model from: %s", model_dir)
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

    logger.info("Loading generation tokenizer for topk_tokens: %s", generation_tokenizer_path)
    generation_tokenizer = AutoTokenizer.from_pretrained(
        generation_tokenizer_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )

    if generation_embed_weight_path:
        generation_weight = load_generation_embedding(generation_model_path, generation_embed_weight_path, device, dtype, logger, local_files_only)
        logger.info("Loaded generation embedding: shape=%s", tuple(generation_weight.shape))
    elif same_path(generation_model_path, guard_model_path):
        generation_weight = guard_model.get_input_embeddings().weight.detach().to(device=device, dtype=dtype)
        logger.info("Using guard input embedding as generation embedding: shape=%s", tuple(generation_weight.shape))
    else:
        generation_weight = load_generation_embedding(generation_model_path, "", device, dtype, logger, local_files_only)
        logger.info("Loaded generation embedding: shape=%s", tuple(generation_weight.shape))

    hidden_size = guard_model.get_input_embeddings().weight.shape[1]
    heads = train_mod.ProbGuardHeads(hidden_size, len(train_mod.CATEGORIES)).to(device=device)
    heads.load_state_dict(metadata["heads"])
    heads.eval()

    template_embeds = train_mod.encode_static_template(tokenizer, guard_model.get_input_embeddings(), device, dtype)
    logger.info("Loaded ProbGuard checkpoint: %s", checkpoint_dir)
    return guard_model, tokenizer, heads, template_embeds, generation_tokenizer, generation_weight, args


def extract_json_object(line: str, field_name: str) -> dict[str, Any] | None:
    marker = f'"{field_name}"'
    start = line.find(marker)
    if start < 0:
        return None
    colon = line.find(":", start + len(marker))
    if colon < 0:
        return None
    brace = line.find("{", colon)
    if brace < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(brace, len(line)):
        char = line[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(line[brace : idx + 1])
    return None


def load_calibration_targets(sample16_path: Path, logger: logging.Logger, limit_records: int = 0) -> dict[int | str, dict[str, float]]:
    targets: dict[int | str, dict[str, float]] = {}
    id_re = re.compile(r'"id"\s*:\s*(".*?"|-?\d+)')
    started_at = time.perf_counter()
    with sample16_path.open("r", encoding="utf-8") as file:
        for row_idx, line in enumerate(tqdm(file, desc=f"targets {sample16_path.name}", dynamic_ncols=True)):
            if limit_records and row_idx >= limit_records:
                break
            match = id_re.search(line)
            if not match:
                continue
            raw_id = match.group(1)
            sample_id: int | str = json.loads(raw_id) if raw_id.startswith('"') else int(raw_id)
            calib = extract_json_object(line, "calibration_probabilities")
            if calib:
                targets[sample_id] = {str(k): float(v) for k, v in calib.items()}
    logger.info("Loaded %d target rows from %s in %.1fs", len(targets), sample16_path, time.perf_counter() - started_at)
    return targets


def valid_steps(steps: Any, k: int) -> bool:
    if not isinstance(steps, list) or len(steps) < k:
        return False
    for step in steps[:k]:
        tokens = step.get("topk_tokens") if isinstance(step, dict) else None
        ids = step.get("topk_token_ids") if isinstance(step, dict) else None
        probs = step.get("topk_probs") if isinstance(step, dict) else None
        has_tokens = isinstance(tokens, list) and bool(tokens) and isinstance(probs, list) and len(tokens) == len(probs)
        has_ids = isinstance(ids, list) and bool(ids) and isinstance(probs, list) and len(ids) == len(probs)
        if not has_tokens and not has_ids:
            return False
    return True


def get_prompt(row: dict[str, Any]) -> str:
    for key in ("goal", "harmful", "prompt"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def compute_ece(preds: list[float], targets: list[float], bins: int) -> float:
    if not preds:
        return 0.0
    pred_array = np.asarray(preds, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (pred_array >= left) & (pred_array <= right)
        else:
            mask = (pred_array >= left) & (pred_array < right)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(pred_array[mask].mean()) - float(target_array[mask].mean()))
    return ece


def summarize_predictions(predictions: list[dict[str, Any]], ece_bins: int) -> dict[str, Any]:
    preds = [float(item["pred_prob"]) for item in predictions]
    targets = [float(item["target_c"]) for item in predictions]
    if not preds:
        return {"samples": 0, "brier": 0.0, "ece": 0.0}

    pred_array = np.asarray(preds, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    binary_target = target_array > 0.5
    binary_pred = pred_array >= 0.5
    summary = {
        "samples": len(predictions),
        "brier": float(np.mean((pred_array - target_array) ** 2)),
        "ece": compute_ece(preds, targets, ece_bins),
        "binary_accuracy_at_0_5": float(np.mean(binary_pred == binary_target)),
        "mean_pred": float(pred_array.mean()),
        "mean_target": float(target_array.mean()),
    }

    category_total = 0
    category_correct = 0
    per_category: dict[str, dict[str, Any]] = {}
    for item in predictions:
        target_category = str(item.get("target_category", "None"))
        pred_category = str(item.get("pred_category", "None"))
        bucket = per_category.setdefault(target_category, {"correct": 0, "total": 0, "accuracy": 0.0})
        bucket["total"] += 1
        category_total += 1
        if pred_category == target_category:
            bucket["correct"] += 1
            category_correct += 1
    for bucket in per_category.values():
        bucket["accuracy"] = float(bucket["correct"] / max(1, bucket["total"]))
    summary["category_accuracy"] = float(category_correct / max(1, category_total))
    summary["category_correct"] = int(category_correct)
    summary["category_total"] = int(category_total)
    summary["per_category_accuracy"] = dict(sorted(per_category.items()))

    per_k: dict[str, Any] = {}
    for k in sorted({int(item["k"]) for item in predictions}):
        subset = [item for item in predictions if int(item["k"]) == k]
        sp = [float(item["pred_prob"]) for item in subset]
        st = [float(item["target_c"]) for item in subset]
        s_pred_array = np.asarray(sp, dtype=np.float64)
        s_target_array = np.asarray(st, dtype=np.float64)
        s_binary_target = s_target_array > 0.5
        s_binary_pred = s_pred_array >= 0.5
        s_category_total = 0
        s_category_correct = 0
        s_per_category: dict[str, dict[str, Any]] = {}
        for item in subset:
            target_category = str(item.get("target_category", "None"))
            pred_category = str(item.get("pred_category", "None"))
            bucket = s_per_category.setdefault(target_category, {"correct": 0, "total": 0, "accuracy": 0.0})
            bucket["total"] += 1
            s_category_total += 1
            if pred_category == target_category:
                bucket["correct"] += 1
                s_category_correct += 1
        for bucket in s_per_category.values():
            bucket["accuracy"] = float(bucket["correct"] / max(1, bucket["total"]))
        per_k[str(k)] = {
            "samples": len(subset),
            "brier": float(np.mean((s_pred_array - s_target_array) ** 2)),
            "ece": compute_ece(sp, st, ece_bins),
            "binary_accuracy_at_0_5": float(np.mean(s_binary_pred == s_binary_target)),
            "category_accuracy": float(s_category_correct / max(1, s_category_total)),
            "category_correct": int(s_category_correct),
            "category_total": int(s_category_total),
            "per_category_accuracy": dict(sorted(s_per_category.items())),
            "mean_pred": float(np.mean(sp)),
            "mean_target": float(np.mean(st)),
        }
    summary["per_k"] = per_k
    return summary


def iter_eval_batches(
    dataset_name: str,
    sample1_path: Path,
    target_map: dict[int | str, dict[str, float]],
    train_mod,
    k_min: int,
    k_max: int,
    k: int,
    k_values: list[int],
    batch_size: int,
    limit_records: int,
):
    batch: list[Any] = []
    current_k_values = k_values if k_values else ([k] if k > 0 else list(range(k_min, k_max + 1)))
    for current_k in current_k_values:
        with sample1_path.open("r", encoding="utf-8") as file:
            desc = f"sample1 {sample1_path.name} k={current_k}"
            for row_idx, line in enumerate(tqdm(file, desc=desc, dynamic_ncols=True)):
                if limit_records and row_idx >= limit_records:
                    break
                row = json.loads(line)
                sample_id = row.get("id", row_idx)
                if sample_id not in target_map:
                    continue
                prompt = get_prompt(row)
                category_label = train_mod.map_category(row.get("primary_category") or row.get("moderation_categories"))
                details = row.get("prefix_generation_details") or {}
                targets = target_map[sample_id]
                k_str = str(current_k)
                if k_str not in targets or k_str not in details:
                    continue
                prefix_samples = details[k_str]
                if not isinstance(prefix_samples, list) or not prefix_samples:
                    continue
                steps = prefix_samples[0]
                if not valid_steps(steps, current_k):
                    continue
                sample = train_mod.ProbGuardSample(
                    prompt=prompt,
                    k=current_k,
                    target_c=float(targets[k_str]),
                    category_label=category_label,
                    steps=steps[:current_k],
                )
                batch.append((dataset_name, sample_id, sample))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
            batch = []


@torch.inference_mode()
def evaluate_dataset(
    dataset_name: str,
    sample1_path: Path,
    sample16_path: Path,
    train_mod,
    model_bundle,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    output_dir: Path | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    guard_model, tokenizer, heads, template_embeds, generation_tokenizer, generation_weight, _ = model_bundle
    target_map = load_calibration_targets(sample16_path, logger, args.limit_records)

    predictions_path = output_dir / f"{dataset_name}_predictions.jsonl" if output_dir is not None else None
    predictions: list[dict[str, Any]] = []
    total_forward_time = 0.0
    total_input_tokens = 0
    total_output_values = 0
    per_k_runtime: dict[int, dict[str, float]] = {}
    started_at = time.perf_counter()
    token_id_cache: dict[str, list[int]] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pred_file = predictions_path.open("w", encoding="utf-8") if predictions_path is not None else None
    try:
        for raw_batch in iter_eval_batches(
            dataset_name=dataset_name,
            sample1_path=sample1_path,
            target_map=target_map,
            train_mod=train_mod,
            k_min=args.k_min,
            k_max=args.k_max,
            k=args.k,
            k_values=args.k_values,
            batch_size=args.batch_size,
            limit_records=args.limit_records,
        ):
            batch_wall_started = time.perf_counter()
            samples = [item[2] for item in raw_batch]
            inputs_embeds, attention_mask, last_indices, target_c, _ = train_mod.build_probguard_batch(
                batch=samples,
                tokenizer=tokenizer,
                guard_embedding=guard_model.get_input_embeddings(),
                template_embeds=template_embeds,
                generation_tokenizer=generation_tokenizer,
                embedding_tokenizer=tokenizer,
                generation_weight=generation_weight,
                device=device,
                model_dtype=dtype,
                max_prompt_len=args.max_prompt_len,
                token_id_cache=token_id_cache,
            )
            total_input_tokens += int(attention_mask.sum().item())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            pred_prob, category_logits = train_mod.forward_probguard(
                guard_model, heads, inputs_embeds, attention_mask, last_indices
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_forward_time += time.perf_counter() - forward_started
            batch_forward_time = time.perf_counter() - forward_started

            pred_prob_cpu = pred_prob.detach().float().cpu().tolist()
            category_cpu = category_logits.argmax(dim=-1).detach().cpu().tolist()
            target_cpu = target_c.detach().float().cpu().tolist()
            batch_input_tokens = int(attention_mask.sum().item())
            batch_wall_time = time.perf_counter() - batch_wall_started
            total_output_values += len(raw_batch) * 2
            for idx, (dataset, sample_id, sample) in enumerate(raw_batch):
                k_runtime = per_k_runtime.setdefault(
                    int(sample.k),
                    {"samples": 0.0, "input_tokens": 0.0, "forward_time_sec": 0.0, "wall_time_sec": 0.0, "output_values": 0.0},
                )
                k_runtime["samples"] += 1.0
                k_runtime["input_tokens"] += batch_input_tokens / max(1, len(raw_batch))
                k_runtime["forward_time_sec"] += batch_forward_time / max(1, len(raw_batch))
                k_runtime["wall_time_sec"] += batch_wall_time / max(1, len(raw_batch))
                k_runtime["output_values"] += 2.0
                item = {
                    "dataset": dataset,
                    "id": sample_id,
                    "k": sample.k,
                    "pred_prob": float(pred_prob_cpu[idx]),
                    "target_c": float(target_cpu[idx]),
                    "pred_category": train_mod.CATEGORIES[int(category_cpu[idx])],
                    "target_category": train_mod.CATEGORIES[int(sample.category_label)],
                }
                predictions.append(item)
                if pred_file is not None:
                    pred_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        if pred_file is not None:
            pred_file.close()

    summary = summarize_predictions(predictions, args.ece_bins)
    wall_time_sec = time.perf_counter() - started_at
    for k_name, item in summary.get("per_k", {}).items():
        runtime = per_k_runtime.get(int(k_name), {})
        samples = max(1.0, float(runtime.get("samples", item.get("samples", 0))))
        wall_time_for_k_sec = float(runtime.get("wall_time_sec", 0.0))
        item.update(
            {
                "input_tokens": int(round(runtime.get("input_tokens", 0.0))),
                "avg_input_tokens_per_sample": float(runtime.get("input_tokens", 0.0) / samples),
                "observed_tokens": int(round(runtime.get("input_tokens", 0.0))),
                "output_values": int(round(runtime.get("output_values", 0.0))),
                "avg_output_values_per_sample": float(runtime.get("output_values", 0.0) / samples),
                "generated_tokens": 0,
                "soft_prefix_steps": int(k_name) * int(samples),
                "forward_time_sec": float(runtime.get("forward_time_sec", 0.0)),
                "avg_forward_ms_per_sample": float(runtime.get("forward_time_sec", 0.0) / samples * 1000.0),
                "wall_time_sec": wall_time_for_k_sec,
                "latency_per_1000_samples_sec": float(wall_time_for_k_sec / samples * 1000.0),
            }
        )
    gpu_peak_alloc_gib = 0.0
    gpu_peak_reserved_gib = 0.0
    gpu_current_alloc_gib = 0.0
    gpu_current_reserved_gib = 0.0
    if device.type == "cuda":
        gpu_peak_alloc_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        gpu_peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
        gpu_current_alloc_gib = torch.cuda.memory_allocated(device) / 1024**3
        gpu_current_reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
    summary.update(
        {
            "dataset": dataset_name,
            "sample1_path": str(sample1_path),
            "sample16_path": str(sample16_path),
            "prediction_path": str(predictions_path) if predictions_path is not None else "",
            "avg_forward_ms_per_sample": (total_forward_time / max(1, len(predictions))) * 1000.0,
            "wall_time_sec": wall_time_sec,
            "latency_per_1000_samples_sec": wall_time_sec / max(1, len(predictions)) * 1000.0,
            "latency_per_1000_samples_ms": wall_time_sec / max(1, len(predictions)) * 1000.0 * 1000.0,
            "forward_time_1000_samples_sec": total_forward_time,
            "input_tokens": int(total_input_tokens),
            "avg_input_tokens_per_sample": float(total_input_tokens / max(1, len(predictions))),
            "generated_tokens": 0,
            "output_values": int(total_output_values),
            "avg_output_values_per_sample": float(total_output_values / max(1, len(predictions))),
            "soft_prefix_steps": int(sum(int(item["k"]) for item in predictions)),
            "observed_tokens": int(total_input_tokens),
            "avg_observed_tokens_per_sample": float(total_input_tokens / max(1, len(predictions))),
            "gpu_peak_alloc_gib": float(gpu_peak_alloc_gib),
            "gpu_peak_reserved_gib": float(gpu_peak_reserved_gib),
            "gpu_current_alloc_gib": float(gpu_current_alloc_gib),
            "gpu_current_reserved_gib": float(gpu_current_reserved_gib),
            "target_rows": len(target_map),
        }
    )
    logger.info(
        "%s | aggregate samples=%d k=%s brier=%.6f ece=%.6f acc=%.4f category_acc=%d/%d(%.4f) mean_pred=%.4f mean_target=%.4f input_tokens=%d generated_tokens=%d probguard_output_values=%d soft_prefix_steps=%d gpu_peak_alloc=%.2fGiB gpu_peak_reserved=%.2fGiB latency_per_1000=%.2fs forward_ms/sample=%.3f",
        dataset_name,
        summary["samples"],
        ",".join(str(item) for item in args.k_values) if args.k_values else (args.k if args.k > 0 else f"{args.k_min}-{args.k_max}"),
        summary["brier"],
        summary["ece"],
        summary["binary_accuracy_at_0_5"],
        summary["category_correct"],
        summary["category_total"],
        summary["category_accuracy"],
        summary["mean_pred"],
        summary["mean_target"],
        summary["input_tokens"],
        summary["generated_tokens"],
        summary["output_values"],
        summary["soft_prefix_steps"],
        summary["gpu_peak_alloc_gib"],
        summary["gpu_peak_reserved_gib"],
        summary["latency_per_1000_samples_sec"],
        summary["avg_forward_ms_per_sample"],
    )
    for k_name, item in summary["per_k"].items():
        logger.info(
            "%s | k=%s samples=%d brier=%.6f ece=%.6f acc=%.4f category_acc=%d/%d(%.4f) mean_pred=%.4f mean_target=%.4f input_tokens=%d generated_tokens=%d probguard_output_values=%d soft_prefix_steps=%d latency_per_1000=%.2fs forward_ms/sample=%.3f",
            dataset_name,
            k_name,
            item["samples"],
            item["brier"],
            item["ece"],
            item["binary_accuracy_at_0_5"],
            item["category_correct"],
            item["category_total"],
            item["category_accuracy"],
            item["mean_pred"],
            item["mean_target"],
            item["input_tokens"],
            item["generated_tokens"],
            item["output_values"],
            item["soft_prefix_steps"],
            item["latency_per_1000_samples_sec"],
            item["avg_forward_ms_per_sample"],
        )
        for category_name, category_item in item["per_category_accuracy"].items():
            logger.info(
                "%s | k=%s category=%s accuracy=%d/%d(%.4f)",
                dataset_name,
                k_name,
                category_name,
                category_item["correct"],
                category_item["total"],
                category_item["accuracy"],
            )
    for category_name, item in summary["per_category_accuracy"].items():
        logger.info(
            "%s | category=%s accuracy=%d/%d(%.4f)",
            dataset_name,
            category_name,
            item["correct"],
            item["total"],
            item["accuracy"],
        )
    return summary


def plot_metric_lines(all_metrics: dict[str, Any], output_dir: Path, logger: logging.Logger) -> None:
    if plt is None:
        logger.warning("matplotlib is not available; skip plotting.")
        return
    datasets = all_metrics.get("datasets", {})
    for metric in ("brier", "ece"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for dataset_name, summary in datasets.items():
            per_k = summary.get("per_k", {})
            k_values = sorted(int(key) for key in per_k.keys())
            values = [float(per_k[str(k)][metric]) for k in k_values]
            ax.plot(k_values, values, marker="o", linewidth=2, label=dataset_name)
        ax.set_xlabel("k")
        ax.set_ylabel(metric.upper() if metric == "ece" else "Brier Score")
        ax.set_title(f"ProbGuard {metric.upper() if metric == 'ece' else 'Brier Score'} by k")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"{metric}_by_k.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Wrote plot: %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained ProbGuard on prefix calibration files.")
    parser.add_argument("--train-script", type=str, default=str(TRAIN_SCRIPT), help="Training script that defines ProbGuard batch/head utilities.")
    parser.add_argument("--checkpoint", type=str, default="all", help='Path to best_checkpoint, alias 8b/4b/0.6b, "auto", or "all".')
    parser.add_argument("--checkpoint-root", type=str, default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-output-files", action="store_true", help="Write predictions, metrics JSON, and plots under --output-dir.")
    parser.add_argument("--log-file", type=str, default="", help="Optional log file path. If empty, log only to stdout.")
    parser.add_argument("--generation-model-override", type=str, default="")
    parser.add_argument("--generation-tokenizer-override", type=str, default="")
    parser.add_argument("--generation-embed-weight-override", type=str, default="")
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer overrides from the local cache only.")
    parser.add_argument("--gpu", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k-values", nargs="*", type=int, default=[5, 10, 15, 20], help="Evaluate these exact prefix lengths. Empty means use --k or --k-min/--k-max.")
    parser.add_argument("--k", type=int, default=10, help="Evaluate exactly one prefix length k. Set <=0 to use --k-min/--k-max.")
    parser.add_argument("--k-min", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_PAIRS.keys()), choices=list(DATASET_PAIRS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_tag = Path(args.train_script).stem.replace("train_single_guard_", "").replace(".", "_")
    run_name = datetime.now().strftime(f"eval_probguard_{train_tag}_%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir) / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    train_mod = load_train_module(Path(args.train_script))

    checkpoint_specs = resolve_checkpoint_specs(args)
    logger = setup_logger(run_output_dir / "launcher.log")
    device = pick_gpu(args.gpu, logger)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if device.type != "cuda":
        dtype = torch.float32

    for checkpoint_name, checkpoint_dir in checkpoint_specs:
        log_path = checkpoint_log_path(args.log_file, run_output_dir, checkpoint_name, len(checkpoint_specs))
        logger = setup_logger(log_path)
        output_dir = run_output_dir / checkpoint_name if args.save_output_files else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Run output directory: %s", run_output_dir)
        logger.info("Checkpoint label: %s", checkpoint_name)
        logger.info("Checkpoint: %s", checkpoint_dir)
        logger.info("Output directory: %s", output_dir if output_dir is not None else "disabled")
        logger.info("Arguments: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))

        model_bundle = load_probguard(
            checkpoint_dir,
            device,
            dtype,
            train_mod,
            logger,
            generation_model_override=args.generation_model_override or None,
            generation_tokenizer_override=args.generation_tokenizer_override or None,
            generation_embed_weight_override=args.generation_embed_weight_override or None,
            local_files_only=args.local_files_only,
        )
        all_metrics = {
            "checkpoint_name": checkpoint_name,
            "checkpoint": str(checkpoint_dir),
            "output_dir": str(output_dir) if output_dir is not None else "",
            "datasets": {},
        }

        try:
            for dataset_name in args.datasets:
                sample1_path, sample16_path = DATASET_PAIRS[dataset_name]
                missing_paths = [path for path in (sample1_path, sample16_path) if not path.exists()]
                if missing_paths:
                    logger.warning("Skip %s because files are missing: %s", dataset_name, ", ".join(str(path) for path in missing_paths))
                    continue
                all_metrics["datasets"][dataset_name] = evaluate_dataset(
                    dataset_name=dataset_name,
                    sample1_path=sample1_path,
                    sample16_path=sample16_path,
                    train_mod=train_mod,
                    model_bundle=model_bundle,
                    device=device,
                    dtype=dtype,
                    args=args,
                    output_dir=output_dir,
                    logger=logger,
                )

            if output_dir is not None:
                metrics_path = output_dir / "metrics_summary.json"
                with metrics_path.open("w", encoding="utf-8") as file:
                    json.dump(all_metrics, file, ensure_ascii=False, indent=2)
                    file.write("\n")
                logger.info("Wrote metrics summary: %s", metrics_path)
                plot_metric_lines(all_metrics, output_dir, logger)
        finally:
            del model_bundle
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
