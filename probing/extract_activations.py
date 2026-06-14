#!/usr/bin/env python3
"""
Direction-cosine probing: layer-wise |cos(∠(v_warm, v_comply))| from contrast sets.

For each variant (base, warm_sft, ours) and each decoder layer ℓ:
    v_warm(ℓ)   = E[h_warm]   - E[h_neutral]
    v_comply(ℓ) = E[h_comply] - E[h_refuse]
and reports |cos(v_warm, v_comply)| per layer to probe_a_cosines.json.
Aggregate across backbones into the paper table with compute_direction_cosine.py.

Requires: torch, transformers, peft, numpy, tqdm.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_prompt_records(path: str) -> List[Dict[str, Any]]:
    """One record per non-empty JSONL line: text plus stable ids for reproducibility."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, str):
                text = row.strip()
            elif "text" in row:
                text = str(row["text"]).strip()
            elif "prompt" in row:
                text = str(row["prompt"]).strip()
            elif "messages" in row and isinstance(row["messages"], list):
                user_parts = [
                    str(m.get("content", ""))
                    for m in row["messages"]
                    if isinstance(m, dict) and m.get("role") == "user"
                ]
                text = (user_parts[-1] if user_parts else "").strip()
            else:
                raise ValueError(f"Unrecognized JSONL record in {path}: {list(row.keys())}")
            if not text:
                continue
            full_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            ext = row.get("id") or row.get("prompt_id")
            pid = str(ext).strip() if ext is not None else full_hash[:24]
            rows.append(
                {
                    "text": text,
                    "prompt_id": pid,
                    "prompt_sha256": full_hash,
                    "line_index": len(rows),
                    **({"messages": row["messages"]} if "messages" in row and isinstance(row["messages"], list) else {}),
                }
            )
    if not rows:
        raise ValueError(f"No prompts read from {path}")
    return rows


def load_contrast_dir(contrast_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Load four JSONLs; return texts for models, full records, and a manifest for reproducibility."""
    classes = ("warm", "neutral", "comply", "refuse")
    texts: Dict[str, List[str]] = {}
    records: Dict[str, List[Dict[str, Any]]] = {}
    files: Dict[str, Any] = {}
    for c in classes:
        p = os.path.join(contrast_dir, f"{c}.jsonl")
        recs = read_prompt_records(p)
        records[c] = recs
        texts[c] = [r["text"] for r in recs]
        files[c] = {"path": p, "n": len(recs), "sha256": sha256_file(p)}
    manifest = {"contrast_dir": os.path.abspath(contrast_dir), "files": files}
    return texts, records, manifest


def validate_contrast_for_paper(
    texts: Dict[str, List[str]],
    paper_strict: bool,
    min_per_class: Optional[int],
) -> None:
    """Enforce equal, sufficiently large class sizes when requested."""
    counts = {k: len(v) for k, v in texts.items()}
    if min_per_class is not None:
        for k, n in counts.items():
            if n < min_per_class:
                raise ValueError(
                    f"Contrast class {k!r} has n={n} < --min-per-class={min_per_class}. "
                    f"Counts: {counts}"
                )
    if paper_strict:
        vals = list(counts.values())
        if len(set(vals)) != 1:
            raise ValueError(
                "Paper mode (--paper-strict) requires equal counts for warm, neutral, comply, refuse. "
                f"Got: {counts}"
            )
        n = vals[0]
        if n < 100:
            raise ValueError(
                f"Paper design expects ≥100 prompts per class; got n={n}."
            )


def try_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def write_run_manifest(out_dir: str, extra: Dict[str, Any]) -> str:
    """Reproducibility sidecar for paper runs."""
    p = os.path.join(out_dir, "run_manifest.json")
    body = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": try_git_commit(),
        **extra,
    }
    save_json(p, body)
    return p


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return obj


def save_json(path: str, obj: Any, atomic: bool = False) -> None:
    """Write JSON. With ``atomic=True``, write to a temp file in the same directory then replace."""
    sanitized = _json_sanitize(obj)
    if not atomic:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False, allow_nan=False)
        return
    dest = os.path.abspath(path)
    d = os.path.dirname(dest) or "."
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json.tmp",
        prefix=os.path.basename(dest) + ".",
        dir=d,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _probe_a_disk_copy(payload: Dict[str, Any], compact_a: bool) -> Dict[str, Any]:
    """Deep copy for disk; optionally strip large tensors (same semantics as in-memory final save)."""
    body = copy.deepcopy(payload)
    if compact_a:
        for v in body["variants"].values():
            v.pop("v_warm", None)
            v.pop("v_comply", None)
            v["vectors_omitted"] = True
    return body


def pick_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n == "float32":
        return torch.float32
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float16", "fp16"):
        return torch.float16
    if n == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    raise ValueError(f"Unknown dtype {name}")


def _tokenizer_pretrained_src(base_model: str, adapter_path: Optional[str]) -> str:
    """Load tokenizer from adapter dir only if it exists and ships HF tokenizer files."""
    if not adapter_path:
        return base_model
    ap = Path(adapter_path)
    if not ap.is_dir():
        return base_model
    if (ap / "tokenizer_config.json").is_file() or (ap / "tokenizer.json").is_file():
        return str(ap)
    return base_model


def load_model_tokenizer(
    base_model: str,
    adapter_path: Optional[str],
    torch_dtype: torch.dtype,
) -> Tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = _tokenizer_pretrained_src(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = dict(
        pretrained_model_name_or_path=base_model,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if not torch.cuda.is_available():
        model = model.to("cpu")
    return model, tokenizer


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """
    HF causal LMs usually expose blocks at ``model.model.layers`` (Llama, Mistral, Qwen, …).
    PEFT may leave an extra ``*ForCausalLM`` shell so layers live at
    ``peft.base_model.model.model.layers`` instead of ``…model.layers``.
    """
    candidates: List[Any] = []
    seen: set[int] = set()

    def push(x: Any) -> None:
        if x is None:
            return
        i = id(x)
        if i in seen:
            return
        seen.add(i)
        candidates.append(x)

    push(model)
    push(getattr(model, "model", None))
    bm = getattr(model, "base_model", None)
    push(bm)
    if bm is not None:
        push(getattr(bm, "model", None))
        inner = getattr(bm, "model", None)
        if inner is not None:
            push(getattr(inner, "model", None))
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
            push(base)
            push(getattr(base, "model", None))
        except Exception:
            pass

    for c in candidates:
        if not isinstance(c, torch.nn.Module):
            continue
        cur: Any = c
        for _ in range(4):
            if cur is None:
                break
            layers = getattr(cur, "layers", None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return layers
            nxt = getattr(cur, "model", None)
            if nxt is cur:
                break
            cur = nxt

    raise AttributeError(
        f"Could not find decoder .layers on {type(model).__name__} "
        "(unsupported or unexpected PEFT/HF wrapper)."
    )


def tokenize_chat_user(
    tokenizer: Any,
    user_text: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_text
    tok_kw: Dict[str, Any] = dict(
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    try:
        enc = tokenizer(prompt, return_token_type_ids=False, **tok_kw)
    except TypeError:
        enc = tokenizer(prompt, **tok_kw)
    return enc


def causal_lm_batch(enc: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Decoder-only models reject e.g. token_type_ids in generate() (Transformers strict check)."""
    keep = ("input_ids", "attention_mask")
    return {k: v for k, v in enc.items() if k in keep and v is not None}


@torch.inference_mode()
def pooled_hidden_all_layers(
    model: torch.nn.Module,
    tokenizer: Any,
    user_text: str,
    max_length: int,
) -> Tuple[np.ndarray, str]:
    """
    Mean-pool hidden states over non-padding positions at each decoder layer.
    Returns array shape [num_layers, hidden_dim] (float32 numpy).
    """
    inputs = causal_lm_batch(tokenize_chat_user(tokenizer, user_text, max_length))
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: embeddings + each layer output
    mask = inputs["attention_mask"].float().unsqueeze(-1)
    mats: List[np.ndarray] = []
    for ell in range(1, len(hs)):
        h = hs[ell]
        masked = h * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (masked.sum(dim=1) / denom)[0]
        vec = pooled.detach().float().cpu().numpy()
        if not np.isfinite(vec).all():
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        mats.append(vec)
    return np.stack(mats, axis=0), "ok"


def _last_assistant_text(messages: Sequence[Dict[str, Any]]) -> str:
    for m in reversed(list(messages)):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", "")).strip()
    return ""


@torch.inference_mode()
def pooled_hidden_assistant_mean_layers(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    max_length: int,
) -> Tuple[np.ndarray, str]:
    """
    Mean-pool hidden states at each decoder layer over **assistant token positions only**
    (last assistant turn in ``messages``). Uses ``offset_mapping`` to align tokens with the
    assistant substring inside the rendered chat template.

    Falls back to mean over all non-padding tokens if no assistant text or no overlapping tokens.
    """
    if not getattr(tokenizer, "chat_template", None):
        mat, st = pooled_hidden_all_layers(model, tokenizer, _last_user_text_from_messages(messages), max_length)
        return mat, f"fallback_no_chat_template:{st}"

    assistant_body = _last_assistant_text(messages)
    if not assistant_body:
        mat, st = pooled_hidden_all_layers(model, tokenizer, _last_user_text_from_messages(messages), max_length)
        return mat, "fallback_empty_assistant"

    prompt = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    astart = prompt.rfind(assistant_body)
    if astart < 0:
        # template may normalize whitespace; try stripped body
        ab2 = " ".join(assistant_body.split())
        astart = prompt.rfind(ab2)
        if astart < 0:
            mat, st = pooled_hidden_all_layers(model, tokenizer, _last_user_text_from_messages(messages), max_length)
            return mat, "fallback_assistant_substring_not_found"
        aend = astart + len(ab2)
    else:
        aend = astart + len(assistant_body)

    try:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
    except TypeError:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    offset_mapping = enc.pop("offset_mapping", None)
    inputs = causal_lm_batch(enc)
    if offset_mapping is None:
        mat, st = pooled_hidden_all_layers(model, tokenizer, _last_user_text_from_messages(messages), max_length)
        return mat, "fallback_no_offsets"

    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    om = offset_mapping[0].tolist()
    am = inputs["attention_mask"][0].tolist()

    mask_1d: List[float] = []
    for t, (cs, ce) in enumerate(om):
        if not am[t]:
            mask_1d.append(0.0)
            continue
        if ce <= cs:
            mask_1d.append(0.0)
            continue
        inter = max(0, min(int(ce), aend) - max(int(cs), astart))
        mask_1d.append(1.0 if inter > 0 else 0.0)

    mask_t = torch.tensor(mask_1d, device=dev, dtype=torch.float32).view(1, -1, 1)
    if float(mask_t.sum()) < 1e-6:
        mat, st = pooled_hidden_all_layers(model, tokenizer, _last_user_text_from_messages(messages), max_length)
        return mat, "fallback_zero_assistant_mask"

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    attn = inputs["attention_mask"].float().unsqueeze(-1)
    eff = mask_t * attn
    mats: List[np.ndarray] = []
    for ell in range(1, len(hs)):
        h = hs[ell]
        masked = h * eff
        denom = eff.sum(dim=1).clamp(min=1e-6)
        pooled = (masked.sum(dim=1) / denom)[0]
        vec = pooled.detach().float().cpu().numpy()
        if not np.isfinite(vec).all():
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        mats.append(vec)
    return np.stack(mats, axis=0), "assistant_span"


def _last_user_text_from_messages(messages: Sequence[Dict[str, Any]]) -> str:
    for m in reversed(list(messages)):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def layer_range_indices(num_layers: int, layer_min: int, layer_max: Optional[int]) -> range:
    hi = num_layers - 1 if layer_max is None else layer_max
    if hi >= num_layers:
        hi = num_layers - 1
    if layer_min < 0 or layer_min > hi:
        raise ValueError(f"Invalid layer range: {layer_min}-{hi} (num_layers={num_layers})")
    return range(layer_min, hi + 1)


@dataclass
class VariantSpec:
    key: str
    adapter_path: Optional[str]


def collect_class_means(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts_by_class: Dict[str, List[str]],
    max_length: int,
    layer_indices: range,
    max_samples_per_class: Optional[int],
    records_by_class: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    assistant_span: bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    """class -> layer -> mean vector (over up to N samples).

    If ``assistant_span`` is True and ``records_by_class[cls][j]`` contains ``messages`` with a
    final assistant turn, use :func:`pooled_hidden_assistant_mean_layers`; otherwise
    :func:`pooled_hidden_all_layers` on the string ``prompts_by_class`` (user-only chat).
    """
    num_layers = len(get_decoder_layers(model))
    li = list(layer_indices)
    if not li:
        raise ValueError("empty layer_indices")
    sums: Dict[str, Dict[int, np.ndarray]] = {c: {i: None for i in li} for c in prompts_by_class}
    counts: Dict[str, Dict[int, int]] = {c: {i: 0 for i in li} for c in prompts_by_class}

    for cls, texts in prompts_by_class.items():
        use = texts if max_samples_per_class is None else texts[: max_samples_per_class]
        recs = (records_by_class or {}).get(cls) or []
        for j, text in enumerate(tqdm(use, desc=f"activations[{cls}]", leave=False)):
            rec = recs[j] if j < len(recs) else {}
            if assistant_span and rec.get("messages"):
                mat, _st = pooled_hidden_assistant_mean_layers(
                    model, tokenizer, list(rec["messages"]), max_length
                )
            else:
                mat, _ = pooled_hidden_all_layers(model, tokenizer, text, max_length)
            for i in li:
                # Accumulate in float64: summing many float32 layer vectors overflows otherwise.
                v = np.asarray(mat[i], dtype=np.float64)
                if sums[cls][i] is None:
                    sums[cls][i] = v.copy()
                else:
                    sums[cls][i] += v
                counts[cls][i] += 1
        for i in li:
            if counts[cls][i] == 0:
                raise RuntimeError(f"No samples for class={cls} layer={i}")
            sums[cls][i] = (sums[cls][i] / float(counts[cls][i])).astype(np.float64, copy=False)
    return sums  # type: ignore[return-value]


def cosine_abs(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    return abs(c)


def run_experiment_a(
    base_model: str,
    variants: List[VariantSpec],
    prompts_by_class: Dict[str, List[str]],
    layer_min: int,
    layer_max: Optional[int],
    max_length: int,
    max_samples_per_class: Optional[int],
    torch_dtype: torch.dtype,
    out_json: str,
    compact_a: bool = False,
    contrast_manifest: Optional[Dict[str, Any]] = None,
    records_by_class: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    assistant_span: bool = False,
) -> Dict[str, Any]:
    num_layers = None
    all_rows: Dict[str, Any] = {}

    for spec in variants:
        label = spec.key
        print(f"[A] variant={label} adapter={spec.adapter_path}", flush=True)
        model, tokenizer = load_model_tokenizer(base_model, spec.adapter_path, torch_dtype)
        nl = len(get_decoder_layers(model))
        num_layers = nl if num_layers is None else num_layers
        lr = layer_range_indices(nl, layer_min, layer_max)

        means = collect_class_means(
            model,
            tokenizer,
            prompts_by_class,
            max_length,
            lr,
            max_samples_per_class,
            records_by_class,
            assistant_span,
        )

        cos_by_layer: Dict[int, float] = {}
        v_warm_by_layer: Dict[int, List[float]] = {}
        v_comply_by_layer: Dict[int, List[float]] = {}
        for i in lr:
            vw = means["warm"][i] - means["neutral"][i]
            vc = means["comply"][i] - means["refuse"][i]
            cos_by_layer[i] = cosine_abs(vw, vc)
            v_warm_by_layer[i] = vw.tolist()
            v_comply_by_layer[i] = vc.tolist()

        all_rows[label] = {
            "adapter_path": spec.adapter_path,
            "cos_abs_by_layer": {str(k): v for k, v in cos_by_layer.items()},
            "v_warm": {str(k): v for k, v in v_warm_by_layer.items()},
            "v_comply": {str(k): v for k, v in v_comply_by_layer.items()},
        }

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Checkpoint after each variant so a long run leaves a recoverable JSON on disk.
        _probe_a_payload_partial = {
            "experiment": "A",
            "base_model": base_model,
            "num_layers": num_layers,
            "layer_min": layer_min,
            "layer_max": layer_max,
            "variants": dict(all_rows),
            "assistant_span": bool(assistant_span),
            "probe_a_complete": False,
            "variants_completed": list(all_rows.keys()),
        }
        if contrast_manifest is not None:
            _probe_a_payload_partial["contrast_manifest"] = contrast_manifest
        save_json(out_json, _probe_a_disk_copy(_probe_a_payload_partial, compact_a), atomic=True)
        print(f"[A] checkpoint wrote {out_json} (variants_done={_probe_a_payload_partial['variants_completed']})", flush=True)

    payload = {
        "experiment": "A",
        "base_model": base_model,
        "num_layers": num_layers,
        "layer_min": layer_min,
        "layer_max": layer_max,
        "variants": all_rows,
        "assistant_span": bool(assistant_span),
        "probe_a_complete": True,
        "variants_completed": list(all_rows.keys()),
    }
    if contrast_manifest is not None:
        payload["contrast_manifest"] = contrast_manifest
    save_json(out_json, _probe_a_disk_copy(payload, compact_a), atomic=True)
    print(f"[A] wrote {out_json}", flush=True)
    if compact_a:
        for v in payload["variants"].values():
            v.pop("v_warm", None)
            v.pop("v_comply", None)
            v["vectors_omitted"] = True
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Direction-cosine probing: |cos(v_warm, v_comply)| per layer -> probe_a_cosines.json"
    )
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B")
    p.add_argument(
        "--adapter-warm-sft",
        required=True,
        help="Checkpoint dir with the generic warmth-FT adapter",
    )
    p.add_argument(
        "--adapter-ours",
        required=True,
        help="Checkpoint dir with our (low-agreeableness + de-escalating) adapter",
    )
    p.add_argument(
        "--contrast-dir",
        required=True,
        help="Directory containing warm.jsonl, neutral.jsonl, comply.jsonl, refuse.jsonl",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", default="auto", help="auto|float32|bfloat16|float16")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-samples-per-class", type=int, default=None)
    p.add_argument("--layer-min", type=int, default=0)
    p.add_argument("--layer-max", type=int, default=None, help="Inclusive; default=all layers")
    p.add_argument(
        "--paper-strict",
        action="store_true",
        help="Require equal class sizes with n>=100.",
    )
    p.add_argument(
        "--min-per-class",
        type=int,
        default=None,
        help="Minimum prompts required per class (weaker than --paper-strict).",
    )
    p.add_argument(
        "--compact-a",
        action="store_true",
        help="Do not store full v_warm/v_comply vectors in probe_a_cosines.json.",
    )
    p.add_argument(
        "--assistant-span",
        action="store_true",
        help=(
            "When a JSONL line has messages[user,assistant], mean-pool hidden states only over "
            "token positions inside the last assistant reply (via offset_mapping)."
        ),
    )
    return p


def cmd_a(args: argparse.Namespace) -> None:
    contrast_dir = args.contrast_dir
    assistant_span = getattr(args, "assistant_span", False)
    prompts_by_class, records_by_class, contrast_manifest = load_contrast_dir(contrast_dir)
    validate_contrast_for_paper(
        prompts_by_class,
        getattr(args, "paper_strict", False),
        getattr(args, "min_per_class", None),
    )
    variants = [
        VariantSpec("base", None),
        VariantSpec("warm_sft", args.adapter_warm_sft),
        VariantSpec("ours", args.adapter_ours),
    ]
    ensure_dir(args.out_dir)
    out_a = os.path.join(args.out_dir, "probe_a_cosines.json")
    run_experiment_a(
        args.base_model,
        variants,
        prompts_by_class,
        args.layer_min,
        args.layer_max,
        args.max_length,
        args.max_samples_per_class,
        pick_dtype(args.dtype),
        out_a,
        compact_a=getattr(args, "compact_a", False),
        contrast_manifest=contrast_manifest,
        records_by_class=records_by_class if assistant_span else None,
        assistant_span=assistant_span,
    )
    write_run_manifest(
        args.out_dir,
        {
            "pipeline": "extract_activations.py",
            "argv": sys.argv,
            "base_model": args.base_model,
            "adapter_warm_sft": args.adapter_warm_sft,
            "adapter_ours": args.adapter_ours,
            "contrast_dir": os.path.abspath(contrast_dir),
            "paper_strict": getattr(args, "paper_strict", False),
            "min_per_class": getattr(args, "min_per_class", None),
            "compact_a": getattr(args, "compact_a", False),
            "assistant_span": assistant_span,
        },
    )


if __name__ == "__main__":
    cmd_a(build_parser().parse_args())
